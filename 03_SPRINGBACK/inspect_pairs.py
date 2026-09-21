# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse,csv,json,os
from collections import defaultdict
import numpy as np
from utils import get_angle,get_sample_id,is_reasonable_file,ensure_dir

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset_dir",required=True); ap.add_argument("--out_dir",default="data/inspect_report"); ap.add_argument("--expected_samples",type=int,default=150); ap.add_argument("--expected_angles",type=int,default=36); args=ap.parse_args(); ensure_dir(args.out_dir)
    files=sorted(os.path.join(args.dataset_dir,f) for f in os.listdir(args.dataset_dir) if f.endswith(".npz")); cov=defaultdict(list); usable=[]; bad={}; rows=[]
    for p in files:
        with np.load(p,allow_pickle=False) as z:
            sid=get_sample_id(z,p); a=get_angle(z,p); n=int(np.asarray(z["X_bend"]).shape[0]); split=str(np.asarray(z["split"]).reshape(-1)[0].decode() if np.asarray(z["split"]).dtype.kind=="S" else np.asarray(z["split"]).reshape(-1)[0]); material=str(np.asarray(z["material_type"]).reshape(-1)[0].decode() if np.asarray(z["material_type"]).dtype.kind=="S" else np.asarray(z["material_type"]).reshape(-1)[0])
        ok,issues=is_reasonable_file(p,{"du_consistency_tol":1e-4,"min_max_du":1e-10,"max_reasonable_du":50.0}); cov[sid].append(int(round(a))); rows.append({"file":p,"sample_id":sid,"angle_deg":a,"split":split,"material_type":material,"n_nodes":n,"status":"OK" if ok else "BAD","issues":";".join(issues)})
        if ok: usable.append(os.path.abspath(p))
        else: bad[p]=issues
    expected=list(range(5,181,5)); coverage_bad={sid:{"have":sorted(set(aa)),"missing":sorted(set(expected)-set(aa))} for sid,aa in cov.items() if sorted(set(aa))!=expected}
    contract_ok=(len(cov)==args.expected_samples and len(files)==args.expected_samples*args.expected_angles and not bad and not coverage_bad)
    print("files={} samples={} usable={} bad={} coverage_bad={}".format(len(files),len(cov),len(usable),len(bad),len(coverage_bad)))
    with open(os.path.join(args.out_dir,"usable_files.txt"),"w") as f:
        for p in usable:f.write(p+"\n")
    if rows:
        with open(os.path.join(args.out_dir,"inspect_summary.csv"),"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out_dir,"inspect_report.json"),"w") as f:
        json.dump({"num_files":len(files),"num_samples":len(cov),"usable":len(usable),"bad":bad,"coverage_bad":coverage_bad,"contract_ok":contract_ok},f,indent=2)
    if not contract_ok:
        raise RuntimeError("pair dataset contract FAILED; reports were written to {}".format(args.out_dir))
    print("PAIR DATASET CONTRACT PASSED: {} x {} = {}".format(args.expected_samples,args.expected_angles,args.expected_samples*args.expected_angles))
if __name__=="__main__": main()
