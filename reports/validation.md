# Local validation

Date: 2026-09-07. Windows 11; Python 3.13.5; Polars 1.44.1; Pandas 2.2.3; NumPy 2.1.3; PyTorch 2.8.0+cpu; Matplotlib 3.10.0.

- `python -m pytest -q`: **11 passed in 16.53 seconds**.
- Coverage: outside-die exclusion, known yield, malformed maps, four-neighbour connectivity, duplicate coordinates, invalid record status, streaming/in-memory equivalence, ViT forward/backward, lot-disjoint splitting, Streamlit initial render and pattern switching.
- `python scripts/benchmark_polars_etl.py --rows 1000000 --repeats 5`: passed output equivalence; five alternating timed repetitions after warmup.
- Final recorded median: Polars **34.70 ms**, Pandas **312.85 ms**; **28.82 million rows/s**, **9.02×** ratio on this synthetic, warm-cache workload. See `benchmark.json` for all samples. Timing variability is substantial; this single local run is illustrative and not an industrial throughput guarantee.
- Inspection PNG generated with the actual plotting code and visually inspected.
- One-epoch end-to-end ViT smoke run on 54 synthetic maps across six synthetic lots completed, including checkpoint reload and held-out evaluation JSON. This verifies execution only; its scores are not WM-811K performance evidence.
- Local combined tests initially aborted with the host Anaconda numeric libraries and PyTorch in one process. A workspace virtual environment with a PyPI NumPy wheel resolved the issue; no unsafe OpenMP override was used.
- Update: real WM-811K source auditing, subset classification and ETL testing are now complete; see [public study](wm811k/README.md). GPU training ran on RTX 4050 with PyTorch 2.5.1/CUDA 11.8 in a separate Python 3.11 environment. The initial CPU-only installation was not evidence of absent GPU hardware.
- Updated test suite: **12 passed in 7.14 seconds**, including public-data deduplication, conflicting-label exclusion and partition checks. The CUDA training/evaluation workflow also completed successfully on actual public data.
- Docker build, Linux CI execution, production deployment and isolated inference-latency benchmarking have not been verified locally.

Use an isolated virtual environment and the supplied requirements to reproduce. Real-data results must be accompanied by the dataset version, split manifest, checkpoint and hardware details.

## Improvement validation — 2026-09-09

- Expanded-data CNN and hybrid training each completed 30 epochs on RTX 4050. Candidate selection used validation only; a separate evaluation command saved selection before audit inference.
- **18 tests passed in 15.42 seconds**, including both encoder gradients, category-preserving augmentation, absent-class metrics, checksum rejection and trained-classifier UI.
- Packaged CPU inference on an original-size historical map matched its recorded GPU prediction. Weights are shipped with a verified SHA-256/model card.
- Training curves and confusion matrices were rendered and visually inspected. Full numerical evidence is in [the improvement study](improvement/README.md).
- The earlier Linux workflow passed after the Streamlit absolute-path test fix. The new workflow result is available in GitHub Actions; local tests above are separately recorded.
