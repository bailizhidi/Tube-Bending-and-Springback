from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from pod_common import list_npz_files, decode_scalar_string, free_tube_mask_from_npz, dataset_signature


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--expect-samples', type=int, default=150)
    ap.add_argument('--expect-frames', type=int, default=181)
    a=ap.parse_args()
    files=list_npz_files(a.data_dir)
    errors=[]; splits=Counter()
    for i,p in enumerate(files,1):
        try:
            with np.load(p,allow_pickle=False) as z:
                req=['tube_mesh_pos','tube_cells','U_tube','node_type','split','D_outer','t_over_D','R_over_D','R_bending']
                miss=[k for k in req if k not in z.files]
                if miss: raise RuntimeError(f'missing {miss}')
                X=np.asarray(z['tube_mesh_pos']); U=np.asarray(z['U_tube'])
                if U.shape!=(a.expect_frames,X.shape[0],3): raise RuntimeError(f'U_tube shape={U.shape}')
                free=free_tube_mask_from_npz(z,X.shape[0])
                split=decode_scalar_string(z['split']).strip().lower(); splits[split]+=1
                if not np.isfinite(U).all(): raise RuntimeError('non-finite U')
                print(f'[{i:03d}/{len(files):03d}] PASS {p.name} split={split} tube={X.shape[0]} free={int(free.sum())}')
        except Exception as e:
            errors.append(f'{p.name}: {e}')
            print(f'[{i:03d}/{len(files):03d}] FAIL {p.name}: {e}')
    if len(files)!=a.expect_samples: errors.append(f'file count {len(files)} != {a.expect_samples}')
    expected=Counter({'train':120,'val':15,'test':15})
    if splits!=expected: errors.append(f'splits {dict(splits)} != {dict(expected)}')
    print('='*100)
    print('DATASET CONTRACT:', 'PASSED' if not errors else 'FAILED')
    print('splits:',dict(splits))
    print('signature:',dataset_signature(files))
    print('errors:',len(errors))
    for e in errors[:20]: print(' -',e)
    print('='*100)
    if errors: raise SystemExit(2)

if __name__=='__main__': main()
