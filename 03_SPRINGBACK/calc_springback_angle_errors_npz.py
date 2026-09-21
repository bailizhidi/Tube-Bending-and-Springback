#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute springback angle errors directly from compact prediction-case NPZ files.

This reuses the validated centerline-ring method from
calc_springback_angle_errors_centerline.py, but avoids exporting hundreds of VTUs.
"""
from __future__ import annotations
import argparse, csv, glob, os, re
from collections import defaultdict
import numpy as np
from calc_springback_angle_errors_centerline import (
    select_two_end_loops, build_graph_adj, bfs_levels_from_loop,
    compute_angle_with_template,
)

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)",str(s))]

def make_template_arrays(points,cells,rings=20,min_loop_nodes=8):
    cells_list=[list(map(int,c)) for c in np.asarray(cells)]
    loop1,loop2,comps=select_two_end_loops(points,cells_list,min_loop_nodes=min_loop_nodes)
    if loop1 is None or loop2 is None:
        raise RuntimeError("cannot identify two end loops; boundary_components={}".format(len(comps)))
    adj=build_graph_adj(cells_list,len(points))
    return {"points_ref":points,"cells":cells_list,"loop1":loop1,"loop2":loop2,
            "levels1":bfs_levels_from_loop(loop1,adj,rings),"levels2":bfs_levels_from_loop(loop2,adj,rings),
            "n_boundary_components":len(comps),"loop1_nodes":len(loop1),"loop2_nodes":len(loop2),"rings":rings}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--case_dir',required=True)
    ap.add_argument('--out_csv',required=True)
    ap.add_argument('--rings',type=int,default=20)
    ap.add_argument('--min_loop_nodes',type=int,default=8)
    ap.add_argument('--min_level_nodes',type=int,default=4)
    ap.add_argument('--min_centroids',type=int,default=3)
    ap.add_argument('--no_nominal_disambiguation',action='store_true')
    args=ap.parse_args()
    files=sorted(glob.glob(os.path.join(args.case_dir,'*.npz')),key=natural_key)
    if not files: raise RuntimeError('no case npz found in {}'.format(args.case_dir))
    rows=[]; failures=[]
    for i,p in enumerate(files,1):
        try:
            with np.load(p,allow_pickle=False) as z:
                sid=int(np.asarray(z['sample_id']).reshape(-1)[0]); nominal=float(np.asarray(z['angle_deg']).reshape(-1)[0])
                xb=np.asarray(z['X_bend'],dtype=np.float64); xt=np.asarray(z['X_spring_true'],dtype=np.float64); xp=np.asarray(z['X_spring_pred'],dtype=np.float64); cells=np.asarray(z['cells'],dtype=np.int64)
            if xt.shape!=xp.shape or xb.shape!=xt.shape: raise RuntimeError('point shape mismatch')
            template=make_template_arrays(xb,cells,args.rings,args.min_loop_nodes)
            nominal_calc=None if args.no_nominal_disambiguation else nominal
            ti=compute_angle_with_template(xt,template,nominal_calc,args.min_level_nodes,args.min_centroids)
            pi=compute_angle_with_template(xp,template,nominal_calc,args.min_level_nodes,args.min_centroids)
            true_angle=float(ti['angle_deg']); pred_angle=float(pi['angle_deg']); err=pred_angle-true_angle
            true_sb=nominal-true_angle; pred_sb=nominal-pred_angle
            rows.append({"sample_id":sid,"nominal_angle_deg":nominal,"true_angle_deg":true_angle,"pred_angle_deg":pred_angle,
                         "angle_error_deg":err,"abs_angle_error_deg":abs(err),"true_springback_deg":true_sb,"pred_springback_deg":pred_sb,
                         "springback_error_deg":pred_sb-true_sb,"abs_springback_error_deg":abs(pred_sb-true_sb),
                         "boundary_components":template['n_boundary_components'],"loop1_nodes":template['loop1_nodes'],"loop2_nodes":template['loop2_nodes'],
                         "n_centroids1":ti['n_centroids1'],"n_centroids2":ti['n_centroids2'],"case_file":os.path.basename(p)})
            if i%50==0 or i==len(files): print('[ANGLE] {}/{}'.format(i,len(files)),flush=True)
        except Exception as e:
            failures.append((p,str(e))); print('[ANGLE FAILED] {} {}'.format(p,e),flush=True)
    os.makedirs(os.path.dirname(args.out_csv) or '.',exist_ok=True)
    fields=["sample_id","nominal_angle_deg","true_angle_deg","pred_angle_deg","angle_error_deg","abs_angle_error_deg","true_springback_deg","pred_springback_deg","springback_error_deg","abs_springback_error_deg","boundary_components","loop1_nodes","loop2_nodes","n_centroids1","n_centroids2","case_file"]
    with open(args.out_csv,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    if failures:
        fp=os.path.splitext(args.out_csv)[0]+'_failures.csv'
        with open(fp,'w',newline='',encoding='utf-8') as f: w=csv.writer(f); w.writerow(['case_file','error']); w.writerows(failures)
    if not rows: raise RuntimeError('all angle cases failed')
    e=np.asarray([r['abs_angle_error_deg'] for r in rows],dtype=float)
    print('angle_cases={} failures={} mean_abs_angle_error_deg={:.8f} max_abs_angle_error_deg={:.8f}'.format(len(rows),len(failures),e.mean(),e.max()))
    by=defaultdict(list)
    for r in rows: by[r['sample_id']].append(r['abs_angle_error_deg'])
    for sid in sorted(by):
        a=np.asarray(by[sid]); print('sample {:04d}: mean={:.6f} max={:.6f} n={}'.format(int(sid),a.mean(),a.max(),len(a)))
if __name__=='__main__': main()
