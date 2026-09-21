from __future__ import annotations
import argparse, json
from pathlib import Path


def ids(path: Path):
    return [int(x.strip()) for x in path.read_text().splitlines() if x.strip()]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workspace',type=Path,default=Path('dataeff_workspace')); args=ap.parse_args()
    s=args.workspace/'subsets'
    a,b,c,d=(set(ids(s/f'train{n}_ids.txt')) for n in (40,60,80,120))
    assert len(a)==40 and len(b)==60 and len(c)==80 and len(d)==120
    assert a < b < c < d
    print('nested subsets: PASS | 40 < 60 < 80 < 120')

if __name__=='__main__': main()
