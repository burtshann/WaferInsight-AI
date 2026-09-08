"""Download the public archive; verify the exact pickle bytes used in this study."""
import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL='https://www.kaggle.com/api/v1/datasets/download/qingyi/wm811k-wafer-map'
EXPECTED='1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65'


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('data/raw'))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    target=a.output/'LSWMD.pkl'
    if target.exists():
        if checksum(target)!=EXPECTED:raise ValueError('Existing source checksum differs; preserve it and choose another output directory.')
        print('Verified existing source:',target)
    else:
        archive=a.output/'wm811k.zip'
        if not archive.exists():
            part=a.output/'wm811k.zip.part'
            with urllib.request.urlopen(URL,timeout=60) as response,part.open('wb') as f:
                shutil.copyfileobj(response,f)
            part.replace(archive)
        with zipfile.ZipFile(archive) as z, z.open('LSWMD.pkl') as source, target.with_suffix('.pkl.part').open('wb') as f:
            shutil.copyfileobj(source,f)
        if checksum(target.with_suffix('.pkl.part'))!=EXPECTED:raise ValueError('Source checksum differs from the benchmark snapshot; do not deserialize.')
        target.with_suffix('.pkl.part').replace(target)
        print('Downloaded and SHA-256 verified:',target)
