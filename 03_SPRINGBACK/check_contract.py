# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, re
import numpy as np
from utils import read_yaml, get_sample_id, get_angle, get_target


def scalar(z, key, default=None):
    if key not in z.files:
        return default
    a=np.asarray(z[key]).reshape(-1)
    if a.size==0:
        return default
    v=a[0]
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
    return v.item() if hasattr(v,'item') else v


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='conf/config_AW_150.yaml')
    ap.add_argument('--pair_dir',required=True)
    args=ap.parse_args()
    cfg=read_yaml(args.config)
    groups=[cfg['train_sample_ids'],cfg['val_sample_ids'],cfg['test_sample_ids']]
    print('split sample counts:', {'train':len(groups[0]),'val':len(groups[1]),'test':len(groups[2])})
    assert list(map(len,groups))==[120,15,15]
    s0,s1,s2=map(set,groups)
    assert not (s0&s1 or s0&s2 or s1&s2)
    assert len(s0|s1|s2)==150

    pair=os.path.abspath(args.pair_dir)
    if not os.path.isdir(pair):
        raise RuntimeError('pair_dir not found: {}'.format(pair))
    files=sorted(os.path.join(pair,f) for f in os.listdir(pair) if f.lower().endswith('.npz'))
    print('pair npz:',len(files))
    assert len(files)==5400, 'expected 5400 NPZ, got {}'.format(len(files))

    coverage={i:set() for i in range(150)}
    required=['X_bend','X_spring','dU_springback','cells','clamp_mask','sample_id','angle_deg','D_outer','R_bending','split','material_type']
    u0_warn=0
    for idx,p in enumerate(files):
        with np.load(p,allow_pickle=False) as z:
            miss=[k for k in required if k not in z.files]
            if miss: raise RuntimeError('{} missing {}'.format(p,miss))
            if 'thickness_value' not in z.files and 'Thickness' not in z.files:
                raise RuntimeError('{} missing thickness_value/Thickness'.format(p))
            sid=get_sample_id(z,p); ang=int(round(get_angle(z,p)))
            if sid not in coverage: raise RuntimeError('bad sample_id {} in {}'.format(sid,p))
            coverage[sid].add(ang)
            mat=str(scalar(z,'material_type',''))
            if 'Base_TC4' not in mat: raise RuntimeError('non-Base_TC4 material in {}: {}'.format(p,mat))
            target_def=str(scalar(z,'target_definition','U_last_minus_U_frame0'))
            if target_def != 'U_last_minus_U_frame0':
                raise RuntimeError('wrong target_definition in {}: {}'.format(p,target_def))
            xb=np.asarray(z['X_bend'],dtype=np.float32); xs=np.asarray(z['X_spring'],dtype=np.float32); du=get_target(z)
            if xb.ndim!=2 or xb.shape[1]!=3 or xs.shape!=xb.shape or du.shape!=xb.shape:
                raise RuntimeError('shape contract failed: {}'.format(p))
            err=float(np.max(np.abs((xs-xb)-du)))
            if err>1e-4: raise RuntimeError('dU consistency failed {} err={}'.format(p,err))
            if 'n_frames' in z.files and int(float(scalar(z,'n_frames',2)))!=2:
                raise RuntimeError('springback ODB was not two-frame in {}'.format(p))
            if 'max_abs_U0' in z.files and float(scalar(z,'max_abs_U0',0.0))>1e-6:
                u0_warn+=1
    expected=set(range(5,181,5))
    bad={sid:sorted(expected-aa) for sid,aa in coverage.items() if aa!=expected}
    if bad: raise RuntimeError('angle coverage failed: {}'.format(bad))
    print('two-frame/U0 diagnostic cases with max_abs_U0>1e-6:',u0_warn)
    print('target definition: dU_springback = U_last - U_frame0')
    print('CONTRACT PASSED')

if __name__=='__main__': main()
