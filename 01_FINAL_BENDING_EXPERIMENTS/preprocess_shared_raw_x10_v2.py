from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from preprocess_graph_utils import (
    build_world_edge_index,
    canonical_split,
    cells_to_mesh_edge_index,
    one_hot_node_type,
    relative_edge4,
    scalar_from_npz,
)

FEATURE_CONTRACT = "ANARESID_X10_shared_raw_packed_v2"
CACHE_FORMAT_VERSION = 4


def decode_scalar(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def cells_and_counts(data) -> Tuple[np.ndarray,np.ndarray]:
    if "cells_global" in data.files:
        cells = np.asarray(data["cells_global"], dtype=np.int64)
    elif "cells" in data.files:
        cells = np.asarray(data["cells"], dtype=np.int64)
    else:
        raise KeyError("NPZ missing cells_global/cells")
    if "tube_cell_num_nodes" in data.files:
        counts = np.asarray(data["tube_cell_num_nodes"], dtype=np.int64).reshape(-1)
    else:
        counts = np.full((cells.shape[0],), cells.shape[1], dtype=np.int64)
    if counts.shape[0] != cells.shape[0]:
        raise RuntimeError("cell_num_nodes length mismatch")
    return cells, counts


def sid_from_npz(path: str) -> int:
    with np.load(path, allow_pickle=False) as z:
        return int(scalar_from_npz(z, ["sample_id","Sample_ID"], -1))


def discover(data_dir: Path):
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files in {data_dir}")
    split_files = {"train":[],"val":[],"test":[]}
    seen=set()
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            sid=int(scalar_from_npz(z,["sample_id","Sample_ID"],-1))
            split=canonical_split(scalar_from_npz(z,["split","Split"],""))
        if sid < 0 or sid in seen:
            raise RuntimeError(f"Invalid/duplicate Sample_ID={sid}")
        seen.add(sid)
        split_files[split].append(str(path))
    for s in split_files:
        split_files[s].sort(key=sid_from_npz)
    return split_files


def contract_signature(world_radius: float, num_time_steps: int) -> str:
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "feature_contract": FEATURE_CONTRACT,
        "world_edge_radius": float(world_radius),
        "num_time_steps": int(num_time_steps),
        "edge_storage": "mesh_edge_attr_static_raw",
        "node_storage": "static_x3_only",
        "normalization_baked_into_cache": False,
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:20]


def source_fingerprint(path: str):
    st=os.stat(path)
    return int(st.st_size), int(st.st_mtime_ns)


def existing_valid(out: Path, source: str, signature: str, T: int):
    if not out.is_file():
        return False
    try:
        p=torch.load(out,map_location="cpu",weights_only=False)
        size,mtime=source_fingerprint(source)
        required=(
            "mesh_pos","world_pos","node_type","is_tube_node","is_tool_node",
            "static_x3","mesh_edge_index","mesh_edge_attr_static_raw",
            "world_edge_indices","cache_signature","feature_contract",
        )
        return (
            all(k in p for k in required)
            and str(p["cache_signature"])==signature
            and str(p["feature_contract"])==FEATURE_CONTRACT
            and int(p["num_time_steps"])==T
            and tuple(p["static_x3"].shape)[1:]==(3,)
            and tuple(p["mesh_edge_attr_static_raw"].shape)[1:]==(4,)
            and len(p["world_edge_indices"])==T-1
            and int(p.get("source_file_size",-1))==size
            and int(p.get("source_mtime_ns",-1))==mtime
        )
    except Exception:
        return False


def build_one(source: str, out: Path, signature: str, T: int, radius: float, device):
    with np.load(source, allow_pickle=False) as z:
        mesh=torch.from_numpy(np.asarray(z["mesh_pos"],dtype=np.float32))
        world=torch.from_numpy(np.asarray(z["world_pos"][:T],dtype=np.float32))
        node_type=torch.from_numpy(np.asarray(z["node_type"],dtype=np.int32)).reshape(-1,1)
        tube=torch.from_numpy(np.asarray(z["is_tube_node"]).reshape(-1).astype(np.bool_))
        tool=torch.from_numpy(np.asarray(z["is_tool_node"]).reshape(-1).astype(np.bool_))
        cells_np, counts_np=cells_and_counts(z)
        cells=torch.from_numpy(cells_np.astype(np.int64))
        counts=torch.from_numpy(counts_np.astype(np.int64))
        sid=int(scalar_from_npz(z,["sample_id","Sample_ID"],-1))
        gid=int(scalar_from_npz(z,["geometry_id","Geometry_ID"],sid))
        split=canonical_split(scalar_from_npz(z,["split","Split"],""))
        region=decode_scalar(scalar_from_npz(z,["region","Region"],"unknown"))
        material=decode_scalar(scalar_from_npz(z,["material_type","Material_Type"],"Base_TC4"))
        D=float(scalar_from_npz(z,["D_outer"],np.nan))
        th=float(scalar_from_npz(z,["Thickness","thickness_value","shell_thickness"],np.nan))
        R=float(scalar_from_npz(z,["R_bending"],np.nan))
        td=float(scalar_from_npz(z,["t_over_D"],th/D))
        rd=float(scalar_from_npz(z,["R_over_D"],R/D))

    if world.shape != (T,mesh.shape[0],3):
        raise RuntimeError(f"sample {sid}: world_pos={tuple(world.shape)}")
    if not torch.isfinite(mesh).all() or not torch.isfinite(world).all():
        raise RuntimeError(f"sample {sid}: NaN/Inf in positions")

    mesh_edge_index=cells_to_mesh_edge_index(cells,counts)
    static_edge_raw=relative_edge4(mesh,mesh_edge_index).float().contiguous()
    static_x3=one_hot_node_type(node_type).float().contiguous()

    mesh_edge_dev=mesh_edge_index.to(device)
    tube_dev=tube.to(device)
    world_edges=[]
    mn=None; mx=0
    for t in tqdm(range(T-1),desc=f"world edges sample {sid:04d}",leave=False):
        cur=world[t].to(device,non_blocking=True)
        idx=build_world_edge_index(cur,mesh_edge_dev,tube_dev,radius=radius)
        ne=int(idx.shape[1]); mn=ne if mn is None else min(mn,ne); mx=max(mx,ne)
        world_edges.append(idx.to("cpu",dtype=torch.int32))
        del cur,idx

    size,mtime=source_fingerprint(source)
    packed={
        "format_version":CACHE_FORMAT_VERSION,
        "cache_signature":signature,
        "feature_contract":FEATURE_CONTRACT,
        "normalization_baked_into_cache":False,
        "source_npz":os.path.basename(source),
        "source_file_size":size,
        "source_mtime_ns":mtime,
        "sample_id":sid,
        "geometry_id":gid,
        "split":split,
        "region":region,
        "material_type":material,
        "D_outer":D,
        "Thickness":th,
        "R_bending":R,
        "t_over_D":td,
        "R_over_D":rd,
        "num_time_steps":T,
        "mesh_pos":mesh.contiguous(),
        "world_pos":world.contiguous(),
        "node_type":node_type.contiguous(),
        "is_tube_node":tube.contiguous(),
        "is_tool_node":tool.contiguous(),
        "cells":cells.contiguous(),
        "cell_num_nodes":counts.contiguous(),
        "static_x3":static_x3,
        "mesh_edge_index":mesh_edge_index.to(dtype=torch.int32).contiguous(),
        "mesh_edge_attr_static_raw":static_edge_raw,
        "world_edge_indices":world_edges,
        "world_edge_count_min":int(mn or 0),
        "world_edge_count_max":int(mx),
    }
    tmp=Path(str(out)+".tmp")
    if tmp.exists(): tmp.unlink()
    torch.save(packed,tmp)
    os.replace(tmp,out)
    return {
        "sample_id":sid,"split":split,"nodes":int(mesh.shape[0]),
        "mesh_edges":int(mesh_edge_index.shape[1]),
        "world_edge_min":int(mn or 0),"world_edge_max":int(mx),
        "file_size_mb":out.stat().st_size/1024**2,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--num-time-steps",type=int,default=181)
    ap.add_argument("--world-radius",type=float,default=2.0)
    ap.add_argument("--overwrite",type=int,choices=[0,1],default=0)
    args=ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for PhysicsNeMo radius_search")
    device=torch.device("cuda:0")
    sf=discover(args.dataset_dir.resolve())
    counts={k:len(v) for k,v in sf.items()}
    if counts != {"train":120,"val":15,"test":15}:
        raise RuntimeError(f"Expected 120/15/15, got {counts}")

    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    sig=contract_signature(args.world_radius,args.num_time_steps)
    summary=[]; split_ids={}

    print("="*100)
    print("ANARESID X10 SHARED RAW PREPROCESSOR V2")
    print("dataset :",args.dataset_dir.resolve())
    print("output  :",out)
    print("splits  :",counts)
    print("contract:",FEATURE_CONTRACT)
    print("signature:",sig)
    print("IMPORTANT: no thickness/velocity/edge normalizer is baked into this cache")
    print("="*100)

    for split in ("train","val","test"):
        names=[]; ids=[]
        for source in sf[split]:
            sid=sid_from_npz(source)
            name=f"{split}_sample_{sid:04d}.pt"
            op=out/name
            names.append(name); ids.append(sid)
            if not args.overwrite and existing_valid(op,source,sig,args.num_time_steps):
                print(f"[SKIP] {name}")
                p=torch.load(op,map_location="cpu",weights_only=False)
                summary.append({
                    "sample_id":sid,"split":split,"status":"SKIPPED_VALID",
                    "nodes":int(p["mesh_pos"].shape[0]),
                    "mesh_edges":int(p["mesh_edge_index"].shape[1]),
                    "world_edge_min":int(p["world_edge_count_min"]),
                    "world_edge_max":int(p["world_edge_count_max"]),
                    "file_size_mb":op.stat().st_size/1024**2,
                })
                continue
            print(f"[PROCESS] {split} sample={sid:04d}")
            r=build_one(source,op,sig,args.num_time_steps,args.world_radius,device)
            r["status"]="OK"; summary.append(r)
            print(
                f"  saved {name} {r['file_size_mb']:.1f} MB "
                f"nodes={r['nodes']} meshE={r['mesh_edges']} "
                f"worldE=[{r['world_edge_min']},{r['world_edge_max']}]"
            )
        (out/f"{split}_manifest.txt").write_text("\n".join(names)+"\n",encoding="utf-8")
        split_ids[split]=ids

    (out/"split_ids.json").write_text(json.dumps({
        "cache_signature":sig,
        "feature_contract":FEATURE_CONTRACT,
        "train_sample_ids":split_ids["train"],
        "val_sample_ids":split_ids["val"],
        "test_sample_ids":split_ids["test"],
    },indent=2),encoding="utf-8")
    (out/"preprocess_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (out/"shared_raw_contract.json").write_text(json.dumps({
        "format_version":CACHE_FORMAT_VERSION,
        "feature_contract":FEATURE_CONTRACT,
        "cache_signature":sig,
        "num_time_steps":args.num_time_steps,
        "world_edge_radius":args.world_radius,
        "split_counts":counts,
        "static_node_features":"static_x3 only",
        "static_edge_features":"mesh_edge_attr_static_raw",
        "normalization_baked_into_cache":False,
    },indent=2),encoding="utf-8")

    print("="*100)
    print("SHARED RAW PREPROCESSING PASSED")
    print("packed:",sum(len(v) for v in sf.values()))
    print("signature:",sig)
    print("="*100)


if __name__=="__main__":
    main()
