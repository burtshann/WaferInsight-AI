from pathlib import Path
import numpy as np
import polars as pl

PATTERNS = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']


def validate_map(values):
    a = np.asarray(values)
    if a.ndim != 2 or min(a.shape) < 2 or max(a.shape) > 512:
        raise ValueError('Expected a 2D wafer map, each dimension between 2 and 512.')
    if not np.isin(a, [0, 1, 2]).all() or not (a > 0).any():
        raise ValueError('Use 0=outside, 1=pass, 2=fail; at least one valid die is required.')
    return a.astype(np.uint8)


def synthetic_map(pattern='Donut', size=64, seed=42):
    if pattern not in PATTERNS:
        raise ValueError('Unknown pattern')
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    r = np.hypot(x, y)
    regions = {'none': r < 0, 'Center': r < .3, 'Donut': (r > .45) & (r < .65),
               'Edge-Loc': (r > .75) & (x > .45), 'Edge-Ring': r > .8,
               'Loc': (x-.35)**2 + (y+.3)**2 < .08, 'Random': r < 2,
               'Scratch': abs(y-.5*x) < .055, 'Near-full': r < 2}
    p = .2 if pattern == 'Random' else .9
    failed = (rng.random(r.shape) < .008) | (regions[pattern] & (rng.random(r.shape) < p))
    return np.where(r <= .95, np.where(failed, 2, 1), 0).astype(np.uint8)


def map_records(a, wafer_id='demo', lot_id='synthetic-lot'):
    a = validate_map(a)
    y, x = np.nonzero(a)
    return pl.DataFrame({'wafer_id': [str(wafer_id)]*len(x), 'lot_id': [str(lot_id)]*len(x),
                         'x': x, 'y': y, 'status': a[y, x]})


def yield_query(frame):
    return (frame.filter(pl.col('status').is_in([1, 2]))
            .group_by(['lot_id', 'wafer_id']).agg(pl.len().alias('total_dies'),
                (pl.col('status') == 1).sum().alias('good_dies'),
                (pl.col('status') == 2).sum().alias('failed_dies'))
            .with_columns((pl.col('good_dies')/pl.col('total_dies')).alias('yield'))
            .sort(['lot_id', 'wafer_id']))


def scan_records(path):
    path = Path(path)
    q = pl.scan_parquet(path) if path.suffix == '.parquet' else pl.scan_csv(path)
    required = {'lot_id', 'wafer_id', 'x', 'y', 'status'}
    if not required.issubset(q.collect_schema().names()):
        raise ValueError(f'Required columns: {sorted(required)}')
    bad = q.select((pl.col('status').is_null() | ~pl.col('status').is_in([0, 1, 2])).any()).collect().item()
    if bad:
        raise ValueError('Invalid or missing die status')
    if q.select(pl.any_horizontal(pl.col(c).is_null() for c in required).any()).collect().item():
        raise ValueError('Missing record fields')
    if q.group_by(['lot_id', 'wafer_id', 'x', 'y']).len().filter(pl.col('len') > 1).limit(1).collect().height:
        raise ValueError('Duplicate die coordinates within wafer')
    return q


def clusters(a, min_size=3):
    """Four-neighbour connected components, bounding boxes in die coordinates."""
    a = validate_map(a)
    seen = set()
    result = []
    for y, x in zip(*np.where(a == 2)):
        if (y, x) in seen:
            continue
        stack, points = [(y, x)], []
        seen.add((y, x))
        while stack:
            cy, cx = stack.pop()
            points.append((cy, cx))
            for ny, nx in [(cy-1,cx), (cy+1,cx), (cy,cx-1), (cy,cx+1)]:
                if 0 <= ny < a.shape[0] and 0 <= nx < a.shape[1] and a[ny,nx] == 2 and (ny,nx) not in seen:
                    seen.add((ny,nx))
                    stack.append((ny,nx))
        if len(points) >= min_size:
            ys, xs = zip(*points)
            result.append({'dies': len(points), 'x_min': int(min(xs)), 'y_min': int(min(ys)),
                           'x_max': int(max(xs)), 'y_max': int(max(ys))})
    return sorted(result, key=lambda c: c['dies'], reverse=True)
