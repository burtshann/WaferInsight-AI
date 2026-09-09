# Improvement protocol — frozen before candidate training

Date: 2026-09-09. No accuracy target is guaranteed.

1. Preserve original complete seed-42 lot allocation and original 809-map validation set. Increase original training-lot samples to caps of 4,000 for none and 2,000 for other classes; use all available rare-class training examples up to the cap. Deduplicate exact original maps and remove conflicting labels exactly as in the initial study.
2. Old 774-map test set is historical comparison only: its scores and confusion matrices have already been inspected. Reserve an additional audit from unused lots in the original complete test allocation, excluding every lot in the previous 4,086-map experiment, capped at 200/class. Report missing classes explicitly; this cannot establish nine-class performance if a class is absent.
3. Compare a spatial CNN and convolutional-stem Transformer trained from scratch, seed42, 30 epochs, batch128, AdamW lr0.001, cosine decay, gradient clipping1.0, label smoothing0.05. Training-only 90-degree rotation/flips and inverse-frequency weighted sampling. No changes to validation/audit images, no augmentation across partitions.
4. Select checkpoint and architecture by validation macro-F1 only. Score the selected model once on the fresh audit after saving selection metadata. Compare frozen original ViT on the same audit. Score selected model on historical test once for continuity. No test-based iteration.
5. Report accuracy, per-class precision/recall/F1, all-nine and supported-class macro-F1, confusion matrices, split/checkpoint hashes and actual GPU environment. Keep existing results intact. A fresh audit with class gaps is additional evidence, not industrial qualification.

Training changes combine architecture, sample count and optimization; gains cannot be attributed to architecture alone. No pretraining is claimed.
