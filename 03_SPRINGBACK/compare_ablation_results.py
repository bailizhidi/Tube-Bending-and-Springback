# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse,csv,os
import numpy as np

def rows(p):
    with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiments',nargs='+',required=True,help='name=prediction_dir'); ap.add_argument('--out_csv',default='predictions/ablation_150_summary.csv'); args=ap.parse_args(); out=[]
    for item in args.experiments:
        name,d=item.split('=',1); m=rows(os.path.join(d,'metrics_test.csv')); mae=np.mean([float(r['mae_mag']) for r in m]); mx=np.max([float(r['max_mag']) for r in m]); am=amax=float('nan'); apath=os.path.join(d,'angle_metrics_centerline.csv')
        if os.path.exists(apath):
            a=rows(apath); vals=[float(r['abs_angle_error_deg']) for r in a]; am=float(np.mean(vals)); amax=float(np.max(vals))
        out.append({'experiment':name,'mean_node_mae_mm':mae,'max_node_error_mm':mx,'mean_angle_error_deg':am,'max_angle_error_deg':amax})
    os.makedirs(os.path.dirname(args.out_csv) or '.',exist_ok=True)
    with open(args.out_csv,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(out[0].keys()));w.writeheader();w.writerows(out)
    for r in out:print(r)
    print(args.out_csv)
if __name__=='__main__':main()
