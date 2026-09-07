"""Equivalent CSV read/filter/group/sort benchmark; validation excluded from timings."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import polars as pl
from waferinsight.core import yield_query, scan_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['synthetic','WM811K'], default='synthetic')
    p.add_argument('--input', type=Path)
    p.add_argument('--rows', type=int, default=1_000_000)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', type=Path, default=Path('reports/benchmark.json'))
    args = p.parse_args()
    if args.rows < 1 or args.repeats < 1:
        p.error('rows and repeats must be positive')
    if args.dataset == 'WM811K' and not args.input:
        p.error('WM811K requires --input exported die-record CSV; dataset is not bundled')
    path = args.input or Path('data/synthetic.csv')
    if not args.input:
        path.parent.mkdir(parents=True, exist_ok=True)
        idx = np.arange(args.rows)
        pl.DataFrame({'lot_id': (idx//100_000).astype(str), 'wafer_id': (idx//4096).astype(str),
                      'x': idx%64, 'y': (idx//64)%64,
                      'status': np.random.default_rng(42).choice([1,2],len(idx),p=[.9,.1])}).write_csv(path)
    if path.suffix != '.csv':
        p.error('Benchmark uses CSV for both engines')
    scan_records(path)  # validation performed once outside timed region
    rows = pl.scan_csv(path).select(pl.len()).collect().item()
    def polars_run():
        return yield_query(pl.scan_csv(path, schema_overrides={'lot_id':pl.String,'wafer_id':pl.String})).collect(engine='streaming')
    def pandas_run():
        d = pd.read_csv(path, dtype={'lot_id':str,'wafer_id':str})
        d = d[d.status.isin([1,2])].assign(good=lambda f:f.status.eq(1), fail=lambda f:f.status.eq(2))
        out = d.groupby(['lot_id','wafer_id']).agg(total_dies=('status','size'),good_dies=('good','sum'),failed_dies=('fail','sum')).reset_index()
        out['yield'] = out.good_dies/out.total_dies
        return out.sort_values(['lot_id','wafer_id']).reset_index(drop=True)
    a,b = polars_run(),pandas_run()
    assert a.select(['lot_id','wafer_id']).rows() == list(b[['lot_id','wafer_id']].itertuples(index=False,name=None))
    np.testing.assert_allclose(a.select(['total_dies','good_dies','failed_dies','yield']).to_numpy(),b[['total_dies','good_dies','failed_dies','yield']].to_numpy())
    samples = {'polars':[], 'pandas':[]}
    for i in range(args.repeats):
        order = [('polars',polars_run),('pandas',pandas_run)]
        for name,fn in order[::1 if i%2 == 0 else -1]:
            start = time.perf_counter()
            fn()
            samples[name].append(time.perf_counter()-start)
    med = {k:statistics.median(v) for k,v in samples.items()}
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    report = {'dataset':args.dataset, 'rows':rows,'sha256':h.hexdigest(),'seconds':samples,
              'median_seconds':med,'polars_rows_per_second':rows/med['polars'],
              'speedup_vs_pandas':med['pandas']/med['polars'], 'equivalence_check':'passed',
              'environment':{'python':platform.python_version(),'os':platform.platform(),
                'cpu':platform.processor(),'polars':pl.__version__,'pandas':pd.__version__, 'polars_threads':pl.thread_pool_size()},
              'method':'warm cache; one warmup per engine; alternating order; CSV read + filter + aggregate + sort; validation excluded'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
