from __future__ import annotations

"""Frozen-checkpoint inference for Geometry-OOD15 and Mesh-Robustness15."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from features import DESIGN_RANGES, design_range_normalize
from model_factory import create_model
from packed_dataset import load_normalization_stats, safe_torch_load, sample_id_from_cache_path
from rollout import rollout_one
from target_space import load_target_stats

OOD15_IDS=[156,157,158,159,162,163,166,167,168,169,171,172,180,181,182]
OOD_LABELS={
    156:"D_low",157:"D_low",158:"D_high",159:"D_high",162:"tD_high",163:"tD_high",
    166:"RD_high",167:"RD_high",168:"D_low+tD_high",169:"D_low+RD_high",
    171:"D_high+RD_high",172:"tD_high+RD_high",180:"D_high_mild",181:"tD_high_mild",182:"RD_high_mild",
}
MESH_GROUPS=[
    (4,183,184),(3,185,186),(139,174,175),(15,187,188),(128,178,179),
]


def _load_external_map(cache_dir: Path) -> Dict[int, Path]:
    files=sorted(cache_dir.glob("external_sample_*.pt")); out={}
    for p in files:
        sid=sample_id_from_cache_path(str(p)); out[sid]=p
    return out


def _load_train_map(cache_dir: Path) -> Dict[int, Path]:
    manifest=cache_dir/"train_manifest.txt"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    out={}
    for name in [x.strip() for x in manifest.read_text().splitlines() if x.strip()]:
        p=cache_dir/name; sid=sample_id_from_cache_path(str(p)); out[sid]=p
    return out


def _prepare_packed(path: Path, device: torch.device):
    p=safe_torch_load(str(path))
    for k in ("mesh_pos","node_type","is_tube_node","is_tool_node","static_x3","mesh_edge_index","mesh_edge_attr_static_raw"):
        p[k]=p[k].to(device,non_blocking=True)
    return p


def _geo_fields(packed: Dict) -> Dict:
    D=float(packed["D_outer"]); td=float(packed["t_over_D"]); rd=float(packed["R_over_D"])
    return {
        "D_outer":D,"t_over_D":td,"R_over_D":rd,
        "D_outer_design_norm":design_range_normalize(D,*DESIGN_RANGES["D_outer"]),
        "t_over_D_design_norm":design_range_normalize(td,*DESIGN_RANGES["t_over_D"]),
        "R_over_D_design_norm":design_range_normalize(rd,*DESIGN_RANGES["R_over_D"]),
    }


def _aggregate(rows: List[Dict]) -> Dict:
    good=[r for r in rows if r.get("status")=="OK"]
    if not good:
        return {"num_cases":0,"failed_cases":len(rows)}
    worst=max(good,key=lambda r:r["mean_free_mm"])
    return {
        "num_cases":len(good),"failed_cases":len(rows)-len(good),
        "mean_free_mm":float(np.mean([r["mean_free_mm"] for r in good])),
        "final_free_mean_mm":float(np.mean([r["final_free_mean_mm"] for r in good])),
        "mean_p95_free_mm":float(np.mean([r["p95_free_mm"] for r in good])),
        "mean_p99_free_mm":float(np.mean([r["p99_free_mm"] for r in good])),
        "max_free_mm":float(max(r["max_free_mm"] for r in good)),
        "worst_case_id":int(worst["eval_sample_id"]),
        "worst_case_mean_free_mm":float(worst["mean_free_mm"]),
        "nan_count":int(sum(r["nan_count"] for r in good)),
    }


def _write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--suite",choices=["ood15","mesh15"],required=True)
    ap.add_argument("--checkpoint",default="best")
    ap.add_argument("--external-cache-dir",type=Path,default=Path("external_workspace/cache_external25"))
    ap.add_argument("--baseline-cache-dir",type=Path,default=Path("dataeff_workspace/cache_views/N120"))
    ap.add_argument("--out-root",type=Path,default=Path("external_results"))
    ap.add_argument("--continue-on-error",type=int,choices=[0,1],default=0)
    a=ap.parse_args()

    cfg=OmegaConf.load(a.config); device=torch.device("cuda:0")
    ck_path=str(cfg.best_checkpoint) if a.checkpoint=="best" else (str(cfg.last_checkpoint) if a.checkpoint=="last" else a.checkpoint)
    ck_path=os.path.abspath(ck_path)
    ts=load_target_stats(str(cfg.target_stats_path),str(cfg.prediction_mode))
    ck=torch.load(ck_path,map_location=device,weights_only=False)
    if str(ck.get("data_fingerprint",""))!=str(ts["raw"].get("data_fingerprint","")):
        raise RuntimeError("checkpoint/Train120 target-stats fingerprint mismatch")
    model=create_model(cfg,device); model.load_state_dict(ck["model"],strict=True); model.eval()
    es=load_normalization_stats(str(cfg.stats_dir))
    if len(es["train_sample_ids"])!=120:
        raise RuntimeError("External evaluation must use frozen Train120 edge statistics")

    ext_dir=a.external_cache_dir.resolve(); contract_path=ext_dir/"external_cache_contract.json"
    contract=json.loads(contract_path.read_text())
    if contract["cache_signature"]!=es["shared_cache_signature"]:
        raise RuntimeError("External cache signature != frozen Train120 edge-stats signature")
    if contract["feature_contract"]!=es["feature_contract"]:
        raise RuntimeError("External feature contract != frozen Train120 feature contract")

    ext=_load_external_map(ext_dir); train=_load_train_map(a.baseline_cache_dir.resolve())
    cases=[]
    if a.suite=="ood15":
        for sid in OOD15_IDS:
            cases.append({"eval_sample_id":sid,"source_sample_id":sid,"mesh_size_mm":1.0,"suite_label":OOD_LABELS[sid],"path":ext[sid]})
    else:
        for src,fine,coarse in MESH_GROUPS:
            if src not in train: raise RuntimeError(f"baseline sample{src:03d} not present in frozen Train120 cache view")
            cases += [
                {"eval_sample_id":fine,"source_sample_id":src,"mesh_size_mm":0.80,"suite_label":"fine","path":ext[fine]},
                {"eval_sample_id":src,"source_sample_id":src,"mesh_size_mm":1.00,"suite_label":"baseline","path":train[src]},
                {"eval_sample_id":coarse,"source_sample_id":src,"mesh_size_mm":1.25,"suite_label":"coarse","path":ext[coarse]},
            ]

    out_dir=(a.out_root/a.suite/str(cfg.experiment_id)).resolve(); out_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    print("="*110)
    print("FROZEN EXTERNAL BENDING EVALUATION")
    print("suite       :",a.suite)
    print("experiment  :",cfg.experiment_id)
    print("checkpoint  :",ck_path)
    print("checkpoint epoch:",int(ck.get("epoch",-1)))
    print("prediction  :",cfg.prediction_mode,"feature:",cfg.feature_mode,"processor:",cfg.processor_type)
    print("cases       :",len(cases))
    print("normalizers : frozen Train120 only")
    print("="*110)

    for i,c in enumerate(cases,1):
        try:
            packed=_prepare_packed(c["path"],device)
            r=rollout_one(model,packed,es["edge_mean"],es["edge_std"],ts,device,cfg,collect_positions=False)
            row={"status":"OK",**{k:v for k,v in c.items() if k!="path"},"cache_file":c["path"].name,**_geo_fields(packed),**r}
            print(f"[{i:02d}/{len(cases):02d}] eval={c['eval_sample_id']:04d} src={c['source_sample_id']:03d} h={c['mesh_size_mm']:.2f} mean={r['mean_free_mm']:.6f} final={r['final_free_mean_mm']:.6f} max={r['max_free_mm']:.6f}",flush=True)
        except Exception as e:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            row={"status":"FAILED",**{k:v for k,v in c.items() if k!="path"},"cache_file":c["path"].name,"error":repr(e)}
            print(f"[{i:02d}/{len(cases):02d}] FAILED eval={c['eval_sample_id']:04d}: {e!r}",flush=True)
            if not a.continue_on_error:
                rows.append(row); _write_csv(out_dir/"case_metrics.csv",rows); raise
        rows.append(row)
        if "packed" in locals():
            del packed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary={
        "suite":a.suite,"experiment_id":str(cfg.experiment_id),"config":str(Path(a.config).resolve()),
        "checkpoint":ck_path,"checkpoint_epoch":int(ck.get("epoch",-1)),
        "processor_type":str(cfg.processor_type),"feature_mode":str(cfg.feature_mode),
        "prediction_mode":str(cfg.prediction_mode),"normalization_source":"frozen Train120 only",
        "aggregate":_aggregate(rows),"rows":rows,
    }
    if a.suite=="mesh15":
        paired=[]
        for src,_,_ in MESH_GROUPS:
            rr={float(r["mesh_size_mm"]):r for r in rows if r.get("status")=="OK" and int(r["source_sample_id"])==src}
            if 1.0 in rr:
                b=rr[1.0]
                for h in (0.8,1.25):
                    if h in rr:
                        x=rr[h]
                        paired.append({
                            "source_sample_id":src,"mesh_size_mm":h,
                            "delta_mean_free_mm":x["mean_free_mm"]-b["mean_free_mm"],
                            "ratio_mean_free":x["mean_free_mm"]/max(b["mean_free_mm"],1e-30),
                            "delta_final_free_mm":x["final_free_mean_mm"]-b["final_free_mean_mm"],
                            "ratio_final_free":x["final_free_mean_mm"]/max(b["final_free_mean_mm"],1e-30),
                            "delta_max_free_mm":x["max_free_mm"]-b["max_free_mm"],
                        })
        summary["paired_vs_1mm"]=paired
        _write_csv(out_dir/"paired_vs_1mm.csv",paired)
    _write_csv(out_dir/"case_metrics.csv",rows)
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\nSUMMARY:",json.dumps(summary["aggregate"],indent=2))
    print("wrote:",out_dir/"summary.json")

if __name__=="__main__": main()
