"""Run one checkpoint in an isolated process, with no future labels on input."""
import argparse
import importlib.metadata
import json
import os
import signal
from pathlib import Path
import threading
import time
import traceback


def timeout_handler(signum, frame):
    raise TimeoutError('Origin inference exceeded its 30-minute budget')


def run(request_path):
    request = json.loads(Path(request_path).read_text())
    settings = request['settings']; output = Path(request['output'])
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='')
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = str(settings['threads'])
    import numpy as np
    import psutil
    process = psutil.Process(); samples = []; stop = threading.Event()
    def monitor():
        while not stop.wait(.01):
            samples.append(process.memory_info().rss)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    results = []; load_seconds = None; start = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(settings['threads'])
        from src.models.foundation import FoundationAdapter
        tic = time.perf_counter()
        adapter = FoundationAdapter(request['spec'], request['checkpoint'], settings)
        load_seconds = time.perf_counter() - tic
        histories = np.load(request['history'], allow_pickle=False)
        for i, origin in enumerate(request['origins']):
            item = {'candidate':request['spec']['id'], 'forecast_origin':origin,
                    'status':'ok', 'error':'', 'fit_seconds':0.0, 'load_seconds':load_seconds if i == 0 else 0.0}
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, settings['worker_timeout_seconds'])
                tic = time.perf_counter(); raw = adapter.predict(histories[i], origin)
                item['inference_seconds'] = time.perf_counter() - tic
                timings = []
                for _ in range(settings['warm_inference_repeats']):
                    tic = time.perf_counter(); repeated = adapter.predict(histories[i], origin)
                    timings.append(time.perf_counter() - tic)
                    if not np.allclose(raw, repeated, rtol=1e-5, atol=1e-5):
                        raise ValueError('Seeded warm forecasts changed')
                np.save(output/f'{i}.npy', raw, allow_pickle=False)
                item.update(warm_inference_median_seconds=float(np.median(timings)),
                            warm_inference_range_seconds=float(np.ptp(timings)),
                            negative_raw_count=int((raw < 0).sum()))
            except Exception as exc:
                item.update(status='timeout' if isinstance(exc,TimeoutError) else 'failed', error=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            results.append(item)
            (output/'resources.json').write_text(json.dumps(results, indent=2)+'\n')
            print(f'{request["spec"]["id"]} {origin}: {item["status"]}', flush=True)
    except Exception as exc:
        traceback.print_exc()
        results = [{'candidate':request['spec']['id'], 'forecast_origin':origin,
                    'status':'failed', 'error':f'{type(exc).__name__}: {exc}', 'fit_seconds':0.0}
                   for origin in request['origins']]
    finally:
        stop.set(); thread.join()
        peak = max(samples + [process.memory_info().rss])
        for item in results:
            item['sampled_peak_rss_bytes'] = peak
        (output/'resources.json').write_text(json.dumps(results, indent=2)+'\n')
        packages = {p:importlib.metadata.version(p) for p in
                    ['numpy','pandas','torch','transformers','chronos-forecasting','timesfm','uni2ts']}
        (output/'environment.json').write_text(json.dumps({'packages':packages,
            'load_seconds':load_seconds, 'elapsed_seconds':time.perf_counter()-start,
            'sampled_peak_rss_bytes':peak}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('request')
    run(parser.parse_args().request)
