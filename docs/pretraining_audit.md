# Stage 6 pretraining and license audit

Audit date: 2026-10-06. Checkpoint README files are downloaded with the immutable revisions and hashed in `configs/stage6_checkpoints.json`. This is a disclosure audit, not proof of training-data absence.

| Checkpoint | Disclosed information | M5-disjoint status |
| --- | --- | --- |
| amazon/chronos-2 | Public card describes real and synthetic pretraining, including Chronos and GiftEval-related sources. The public Chronos dataset collection contains M5; membership alone does not prove this checkpoint trained on it. | Unresolved; possible overlap. Do not claim clean out-of-domain M5 evaluation. |
| google/timesfm-2.5-200m-pytorch | Card describes broad pretraining including public and synthetic sources. A complete example-level exclusion of this pilot's dates/series has not been established here. | Unresolved; do not treat lack of a named M5 reference as proof of exclusion. |
| Salesforce/moirai-1.1-R-small | The original Moirai paper, Table 11, lists M5 in LOTSA (30,490 series; 58,327,370 observations). The official LOTSA builder puts M5 in its pretraining group and uses the GluonTS test collection, which includes historical targets. | Published family-corpus overlap: treat as contaminated for clean M5 generalization claims. Exact 1.1 checkpoint example-level lineage is not certified. |

A model can be zero-shot **with respect to this project's fitting procedure** while having encountered related or identical data before release. These experiments cannot establish clean zero-shot generalization to unseen M5 solely from their date split. Keep this limitation in all final claims; an independently verified disjoint dataset would be needed to resolve it.

The reported M5 observation count corresponds to 1,913 days per series. **Inference:** this spans our validation dates if the published corpus follows the standard M5 series alignment. We have not independently matched individual training examples to the checkpoint. The evidence is substantially stronger than an absence-of-disclosure concern.

## Sources and extraction checks

- [Chronos-2 model card](https://huggingface.co/amazon/chronos-2).
- [Chronos dataset collection](https://huggingface.co/datasets/autogluon/chronos_datasets): M5 is present in this collection; this is not checkpoint-specific training evidence.
- [Chronos-2 pipeline source](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py): explicitly returns median as the second `predict_quantiles` result.
- [TimesFM 2.5 model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch) and [official code](https://github.com/google-research/timesfm).
- [Moirai checkpoint card](https://huggingface.co/Salesforce/moirai-1.1-R-small) and [official Uni2TS implementation](https://github.com/SalesforceAIResearch/uni2ts).

Chronos-2 and TimesFM selected cards specify Apache-2.0. Moirai 1.1's card specifies CC-BY-NC-4.0, so this checkpoint is not an unrestricted commercial deployment option. Package licenses do not override checkpoint licenses. Keep downloaded pinned cards as the version-specific license evidence.

- [Original Moirai paper, Table 11](https://arxiv.org/pdf/2402.02592): M5 membership in LOTSA.
- [Official LOTSA GluonTS builder](https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/data/builder/lotsa_v1/gluonts.py): M5 in `PRETRAIN_GROUP`, loaded from `.test`; repository evidence is not a checkpoint-specific training log.
