# Local validation

Date: 2026-09-07. Windows 11; Python 3.13.5; Polars 1.44.1; Pandas 2.2.3; NumPy 2.1.3; PyTorch 2.8.0+cpu; Matplotlib 3.10.0.

- `python -m pytest -q`: **11 passed in 16.53 seconds**.
- Coverage: outside-die exclusion, known yield, malformed maps, four-neighbour connectivity, duplicate coordinates, invalid record status, streaming/in-memory equivalence, ViT forward/backward, lot-disjoint splitting, Streamlit initial render and pattern switching.
- `python scripts/benchmark_polars_etl.py --rows 1000000 --repeats 5`: passed output equivalence; five alternating timed repetitions after warmup.
- Final recorded median: Polars **34.70 ms**, Pandas **312.85 ms**; **28.82 million rows/s**, **9.02×** ratio on this synthetic, warm-cache workload. See `benchmark.json` for all samples. Timing variability is substantial; this single local run is illustrative and not an industrial throughput guarantee.
- Inspection PNG generated with the actual plotting code and visually inspected.
- One-epoch end-to-end ViT smoke run on 54 synthetic maps across six synthetic lots completed, including checkpoint reload and held-out evaluation JSON. This verifies execution only; its scores are not WM-811K performance evidence.
- Local combined tests initially aborted with the host Anaconda numeric libraries and PyTorch in one process. A workspace virtual environment with a PyPI NumPy wheel resolved the issue; no unsafe OpenMP override was used.
- Docker build, Linux CI execution, real WM-811K training, deployment and GPU inference were not validated locally.

Use an isolated virtual environment and the supplied requirements to reproduce. Real-data results must be accompanied by the dataset version, split manifest, checkpoint and hardware details.
