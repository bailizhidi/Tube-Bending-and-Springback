# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Sequence
import torch
from tqdm import tqdm
from springback_dataset import RunningStats, build_raw_data
from utils import discover_npz_files, ensure_dir, filter_reasonable_files, read_yaml, save_json, split_by_sample_id, write_text_lines

def graph_to_dict(g, source):
    d={k:(v.cpu() if torch.is_tensor(v) else v) for k,v in g.__dict__.items() if k not in {"batch","ptr","num_graphs"}}
    d["source_npz"]=os.path.abspath(source); return d

def convert_one(task):
    npz_path, pt_path, feature_set, overwrite=task
    try:
        if os.path.exists(pt_path) and not overwrite: return npz_path,pt_path,False,""
        g=build_raw_data(npz_path,feature_set=feature_set); tmp=pt_path+".tmp"; torch.save(graph_to_dict(g,npz_path),tmp); os.replace(tmp,pt_path)
        return npz_path,pt_path,True,""
    except Exception as e: return npz_path,pt_path,False,str(e)

def compute_stats(files: Sequence[str]):
    first=torch.load(files[0],map_location="cpu",weights_only=False)
    ns=RunningStats(first["x"].shape[1]); es=RunningStats(first["edge_attr"].shape[1]); ys=RunningStats(first["y"].shape[1])
    for f in tqdm(files,desc="stats"):
        d=torch.load(f,map_location="cpu",weights_only=False); ns.update(d["x"]); es.update(d["edge_attr"]); ys.update(d["y"])
    nm,nstd=ns.finalize(); em,estd=es.finalize(); ym,ystd=ys.finalize()
    return {"node_mean":nm,"node_std":nstd,"edge_mean":em,"edge_std":estd,"y_mean":ym,"y_std":ystd,
            "num_node_features":torch.tensor([first["x"].shape[1]]),"num_edge_features":torch.tensor([first["edge_attr"].shape[1]]),"num_output_features":torch.tensor([first["y"].shape[1]])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--data_dir"); ap.add_argument("--usable_files"); ap.add_argument("--processed_dir"); ap.add_argument("--num_workers",type=int,default=8); ap.add_argument("--overwrite",action="store_true"); args=ap.parse_args()
    cfg=read_yaml(args.config)
    if args.data_dir: cfg["data_dir"]=args.data_dir
    if args.usable_files: cfg["usable_files"]=args.usable_files
    if args.processed_dir: cfg["processed_dir"]=args.processed_dir
    out=cfg["processed_dir"]; graph_dir=os.path.join(out,"graphs"); split_dir=os.path.join(out,"splits"); ensure_dir(graph_dir); ensure_dir(split_dir)
    files=discover_npz_files(cfg["data_dir"],cfg.get("usable_files","")); files,bad=filter_reasonable_files(files,cfg); save_json(bad,os.path.join(out,"rejected.json"))
    print("pair files usable:",len(files),"rejected:",len(bad))
    splits=split_by_sample_id(files,cfg)
    expected={"train":int(cfg.get("expected_train_graphs",4320)),"val":int(cfg.get("expected_val_graphs",540)),"test":int(cfg.get("expected_test_graphs",540))}
    for k,fs in splits.items():
        print(k,len(fs),"expected",expected[k]);
        if len(fs)!=expected[k]: raise RuntimeError("{} graph count mismatch: {} != {}".format(k,len(fs),expected[k]))
        write_text_lines(fs,os.path.join(split_dir,k+"_npz_files.txt"))
    all_npz=sorted(set(sum(splits.values(),[]))); feature=cfg.get("feature_set","geom")
    mapping={os.path.abspath(f):os.path.abspath(os.path.join(graph_dir,os.path.splitext(os.path.basename(f))[0]+".pt")) for f in all_npz}
    tasks=[(f,mapping[os.path.abspath(f)],feature,args.overwrite) for f in all_npz]; errors={}; converted=existing=0
    if args.num_workers<=1:
        iterator=map(convert_one,tasks)
        for a,b,c,e in tqdm(iterator,total=len(tasks),desc="npz->pt"):
            if e: errors[a]=e
            elif c: converted+=1
            else: existing+=1
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futs=[ex.submit(convert_one,t) for t in tasks]
            for fut in tqdm(as_completed(futs),total=len(futs),desc="npz->pt"):
                a,b,c,e=fut.result()
                if e: errors[a]=e
                elif c: converted+=1
                else: existing+=1
    save_json(errors,os.path.join(out,"conversion_errors.json"))
    if errors: raise RuntimeError("conversion errors: {}".format(len(errors)))
    splits_pt={k:[mapping[os.path.abspath(f)] for f in fs] for k,fs in splits.items()}
    for k,fs in splits_pt.items(): write_text_lines(fs,os.path.join(split_dir,k+"_files.txt"))
    stats=compute_stats(splits_pt["train"]); torch.save(stats,os.path.join(out,"stats.pt"))
    meta={"cfg":cfg,"splits_npz":splits,"splits_pt":splits_pt,"num_train":len(splits_pt["train"]),"num_val":len(splits_pt["val"]),"num_test":len(splits_pt["test"]),"converted":converted,"existing":existing}
    save_json(meta,os.path.join(out,"processed_meta.json"))
    print("DONE node_dim={} edge_dim={} out_dim={}".format(int(stats["num_node_features"].item()),int(stats["num_edge_features"].item()),int(stats["num_output_features"].item())))
if __name__=="__main__": main()
