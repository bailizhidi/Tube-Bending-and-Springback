# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse,csv,os
import numpy as np
import torch
from model import MeshGraphNetOneStep
from springback_dataset import build_raw_data, denormalize_y
from export_vtu import write_vtu
from utils import ensure_dir

def load_model(path,device):
    ck=torch.load(path,map_location=device,weights_only=False); cfg=ck["cfg"]; st={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in ck["stats"].items()}
    m=MeshGraphNetOneStep(int(st["num_node_features"].item()),int(st["num_edge_features"].item()),int(st["num_output_features"].item()),int(cfg.get("hidden_dim",128)),int(cfg.get("num_message_passing_steps",12)),int(cfg.get("mlp_layers",2)),float(cfg.get("dropout",0.0))).to(device)
    m.load_state_dict(ck["model_state"]); m.eval(); return m,cfg,st,ck

def norm_data(g,st):
    g.x_raw=g.x.clone(); g.edge_attr_raw=g.edge_attr.clone(); g.y_raw=g.y.clone(); g.x=(g.x-st["node_mean"].cpu())/st["node_std"].cpu(); g.edge_attr=(g.edge_attr-st["edge_mean"].cpu())/st["edge_std"].cpu(); g.y=(g.y-st["y_mean"].cpu())/st["y_std"].cpu(); return g

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--split",default="test",choices=["train","val","test"]); ap.add_argument("--out_dir",required=True); ap.add_argument("--export_vtu",action="store_true"); ap.add_argument("--max_export",type=int,default=30); ap.add_argument("--save_case_npz",action="store_true"); args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model,cfg,st,ck=load_model(args.ckpt,device); ensure_dir(args.out_dir); files=ck["splits"][args.split]; rows=[]; count=0
    for i,path in enumerate(files,1):
        g=build_raw_data(path,feature_set=cfg.get("feature_set","geom")); g=norm_data(g,{k:(v.cpu() if torch.is_tensor(v) else v) for k,v in st.items()}); gd=g.to(device); pred=denormalize_y(model(gd),st).cpu().numpy(); true=g.y_raw.numpy(); xb=g.x_bend.numpy(); xs=g.x_spring.numpy(); xp=xb+pred; err=pred-true; mag=np.linalg.norm(err,axis=1); sid=int(g.sample_id.item()); ang=float(g.angle_deg.item())
        rows.append({"file":path,"sample_id":sid,"angle_deg":ang,"mae_dx":float(np.mean(np.abs(err[:,0]))),"mae_dy":float(np.mean(np.abs(err[:,1]))),"mae_dz":float(np.mean(np.abs(err[:,2]))),"mae_mag":float(mag.mean()),"p95_mag":float(np.percentile(mag,95)),"max_mag":float(mag.max())})
        base="sample_{:04d}_angle_{:03d}".format(sid,int(round(ang))); cells=g.cells.numpy()
        if args.save_case_npz:
            case_dir=os.path.join(args.out_dir,"cases_npz"); ensure_dir(case_dir)
            np.savez_compressed(os.path.join(case_dir,base+".npz"),sample_id=np.asarray([sid],dtype=np.int32),angle_deg=np.asarray([ang],dtype=np.float32),X_bend=xb.astype(np.float32),X_spring_true=xs.astype(np.float32),X_spring_pred=xp.astype(np.float32),dU_true=true.astype(np.float32),dU_pred=pred.astype(np.float32),cells=cells.astype(np.int32))
        if args.export_vtu and count<args.max_export:
            pd={"dU_true":true,"dU_pred":pred,"dU_error":err,"error_mag":mag}
            write_vtu(os.path.join(args.out_dir,base+"_GT_bend.vtu"),xb,cells,pd); write_vtu(os.path.join(args.out_dir,base+"_GT_spring.vtu"),xs,cells,pd); write_vtu(os.path.join(args.out_dir,base+"_PRED_spring.vtu"),xp,cells,pd); count+=1
        if i%50==0 or i==len(files): print("pred {}/{}".format(i,len(files)),flush=True)
    csvp=os.path.join(args.out_dir,"metrics_{}.csv".format(args.split));
    with open(csvp,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    mae=float(np.mean([r["mae_mag"] for r in rows])); p95=float(np.mean([r["p95_mag"] for r in rows])); mx=float(np.max([r["max_mag"] for r in rows])); print("[TEST] graphs={} mean_node_mae_mm={:.6e} mean_case_p95_mm={:.6e} worst_node_mm={:.6e}".format(len(rows),mae,p95,mx)); print(csvp)
if __name__=="__main__": main()
