# Stage 5 environment setup

Python packages are pinned in `requirements.txt`. On this Apple Silicon Mac, importing XGBoost 3.4.1 initially failed because `libomp.dylib` was absent. Installing Homebrew **libomp 23.1.2** resolved the import failure; XGBoost reports OpenMP enabled and CUDA disabled.

For a new macOS environment with Homebrew already installed, install the native prerequisite before running the Python tests:

```bash
HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install libomp
python -m pip install -r requirements.txt
```

`requirements.txt` alone does not install this native macOS dependency. Other platforms may require their own compiler/OpenMP runtime packages.

## Installation record and side effect

During this session, the approved `brew install libomp` command also ran Homebrew's default auto-update and automatic cleanup. That cleanup was not separately requested. Its output reported removing Node 25.2.1 and mongosh 2.5.10, plus dependencies it classified as unneeded (including ICU, brotli, c-ares, libuv, and HTTP/QUIC-related libraries), as well as cached downloads. A subsequent read-only package inventory still listed Node 23.9.0 and mongosh 2.4.0. That inventory does not establish that those older installations remain functional. No attempt was made to upgrade, restore, or relink unrelated applications as part of the forecasting task.

The setup command above disables those automatic Homebrew behaviors for future reproduction. XGBoost import and the project tests passed after libomp installation.
