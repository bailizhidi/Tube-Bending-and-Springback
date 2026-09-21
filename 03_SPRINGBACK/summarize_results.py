# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse,csv,os
import numpy as np

def read(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pred_dir',required=True); args=ap.parse_args(); mp=os.path.join(args.pred_dir,'metrics_test.csv'); apath=os.path.join(args.pred_dir,'angle_metrics_centerline.csv'); m=read(mp)
    mae=np.asarray([float(r['mae_mag']) for r in m]); p95=np.asarray([float(r['p95_mag']) for r in m]); mx=np.asarray([float(r['max_mag']) for r in m]); print('graphs              :',len(m)); print('Mean node MAE (mm)  : {:.8f}'.format(mae.mean())); print('Mean graph p95 (mm) : {:.8f}'.format(p95.mean())); print('Max node error (mm) : {:.8f}'.format(mx.max()))
    if os.path.exists(apath):
        a=read(apath); e=np.asarray([float(r['abs_angle_error_deg']) for r in a]); print('Angle cases         :',len(a)); print('Mean angle err (deg): {:.8f}'.format(e.mean())); print('Max angle err (deg) : {:.8f}'.format(e.max()))
if __name__=='__main__':main()
