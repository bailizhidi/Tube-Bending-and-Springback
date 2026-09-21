from __future__ import annotations

"""Build inference-only packed cache for the frozen External25 dataset.

Important contract:
- External data are NEVER used to fit edge/target normalization statistics.
- The cache uses the exact same static graph/feature contract as Train120.
- Dynamic world edges are rebuilt from the predicted state during rollout, exactly
  as in the frozen Test15 inference path, so no GT future world edges are cached.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from packed_dataset import load_normalization_stats
from preprocess_graph_utils import (
    cells_to_mesh_edge_index,
    one_hot_node_type,
    relative_edge4,
    scalar_from_npz,
)
from preprocess_shared_raw_x10_v2 import (
    CACHE_FORMAT_VERSION,
    FEATURE_CONTRACT,
    cells_and_counts,
    contract_signature,
    decode_scalar,
    source_fingerprint,
)

OOD15_IDS = [156,157,158,159,162,163,166,167,168,169,171,172,180,181,182]
MESH10_IDS = [174,175,178,179,183,184,185,186,187,188]
EXPECTED25 = set(OOD15_IDS + MESH10_IDS)
OOD_LABELS = {
    156:"D_low",157:"D_low",158:"D_high",159:"D_high",
    162:"tD_high",163:"tD_high",166:"RD_high",167:"RD_high",
    168:"D_low+tD_high",169:"D_low+RD_high",171:"D_high+RD_high",
    172:"tD_high+RD_high",180:"D_high_mild",181:"tD_high_mild",182:"RD_high_mild",
}
MESH_MAP = {
    174:(139,0.80),175:(139,1.25),178:(128,0.80),179:(128,1.25),
    183:(4,0.80),184:(4,1.25),185:(3,0.80),186:(3,1.25),
    187:(15,0.80),188:(15,1.25),
}


def _sid(path: Path) -> int:
    with np.load(path, allow_pickle=False) as z:
        return int(scalar_from_npz(z,["sample_id","Sample_ID"],-1))


def _discover(dataset_dir: Path) -> Dict[int, Path]:
    files=sorted(dataset_dir.glob("*.npz"))
    if len(files)!=25:
        raise RuntimeError(f"Expected 25 external NPZs, got {len(files)} in {dataset_dir}")
    out={}
    for p in files:
        sid=_sid(p)
        if sid in out: raise RuntimeError(f"duplicate external Sample_ID={sid}")
        out[sid]=p
    if set(out)!=EXPECTED25:
        raise RuntimeError(
            f"External25 ID mismatch; missing={sorted(EXPECTED25-set(out))}, "
            f"extra={sorted(set(out)-EXPECTED25)}"
        )
    return out


def _existing_valid(out: Path, source: Path, signature: str, T: int) -> bool:
    if not out.is_file(): return False
    try:
        p=torch.load(out,map_location="cpu",weights_only=False)
        size,mtime=source_fingerprint(str(source))
        required=("mesh_pos","world_pos","node_type","is_tube_node","is_tool_node",
                  "static_x3","mesh_edge_index","mesh_edge_attr_static_raw",
                  "cache_signature","feature_contract")
        return (
            all(k in p for k in required)
            and str(p["cache_signature"])==signature
            and str(p["feature_contract"])==FEATURE_CONTRACT
            and int(p["num_time_steps"])==T
            and int(p.get("source_file_size",-1))==size
            and int(p.get("source_mtime_ns",-1))==mtime
        )
    except Exception:
        return False


def _build_one(source: Path, out: Path, signature: str, T: int) -> Dict:
    with np.load(source,allow_pickle=False) as z:
        mesh=torch.from_numpy(np.asarray(z["mesh_pos"],dtype=np.float32))
        world=torch.from_numpy(np.asarray(z["world_pos"][:T],dtype=np.float32))
        node_type=torch.from_numpy(np.asarray(z["node_type"],dtype=np.int32)).reshape(-1,1)
        tube=torch.from_numpy(np.asarray(z["is_tube_node"]).reshape(-1).astype(np.bool_))
        tool=torch.from_numpy(np.asarray(z["is_tool_node"]).reshape(-1).astype(np.bool_))
        cells_np,counts_np=cells_and_counts(z)
        cells=torch.from_numpy(cells_np.astype(np.int64))
        counts=torch.from_numpy(counts_np.astype(np.int64))
        sid=int(scalar_from_npz(z,["sample_id","Sample_ID"],-1))
        gid=int(scalar_from_npz(z,["geometry_id","Geometry_ID"],sid))
        raw_split=decode_scalar(scalar_from_npz(z,["split","Split"],"external"))
        region=decode_scalar(scalar_from_npz(z,["region","Region"],"external"))
        material=decode_scalar(scalar_from_npz(z,["material_type","Material_Type"],"Base_TC4"))
        D=float(scalar_from_npz(z,["D_outer"],np.nan))
        th=float(scalar_from_npz(z,["Thickness","thickness_value","shell_thickness"],np.nan))
        R=float(scalar_from_npz(z,["R_bending"],np.nan))
        td=float(scalar_from_npz(z,["t_over_D"],th/D))
        rd=float(scalar_from_npz(z,["R_over_D"],R/D))

    if world.shape!=(T,mesh.shape[0],3):
        raise RuntimeError(f"sample {sid}: world_pos={tuple(world.shape)}")
    if not torch.isfinite(mesh).all() or not torch.isfinite(world).all():
        raise RuntimeError(f"sample {sid}: NaN/Inf in positions")
    if "Base_TC4" not in material:
        raise RuntimeError(f"sample {sid}: material={material!r}, expected Base_TC4")

    mesh_edge_index=cells_to_mesh_edge_index(cells,counts)
    static_edge_raw=relative_edge4(mesh,mesh_edge_index).float().contiguous()
    static_x3=one_hot_node_type(node_type).float().contiguous()
    size,mtime=source_fingerprint(str(source))

    packed={
        "format_version":CACHE_FORMAT_VERSION,
        "external_cache_format":"ANARESID_external_inference_cache_v1",
        "cache_signature":signature,
        "feature_contract":FEATURE_CONTRACT,
        "normalization_baked_into_cache":False,
        "source_npz":source.name,
        "source_file_size":size,"source_mtime_ns":mtime,
        "sample_id":sid,"geometry_id":gid,"split":raw_split,"region":region,
        "material_type":material,"D_outer":D,"Thickness":th,"R_bending":R,
        "t_over_D":td,"R_over_D":rd,"num_time_steps":T,
        "mesh_pos":mesh.contiguous(),"world_pos":world.contiguous(),
        "node_type":node_type.contiguous(),"is_tube_node":tube.contiguous(),
        "is_tool_node":tool.contiguous(),"cells":cells.contiguous(),
        "cell_num_nodes":counts.contiguous(),"static_x3":static_x3,
        "mesh_edge_index":mesh_edge_index.to(dtype=torch.int32).contiguous(),
        "mesh_edge_attr_static_raw":static_edge_raw,
    }
    tmp=Path(str(out)+".tmp")
    if tmp.exists(): tmp.unlink()
    torch.save(packed,tmp); os.replace(tmp,out)
    return {
        "sample_id":sid,"nodes":int(mesh.shape[0]),"mesh_edges":int(mesh_edge_index.shape[1]),
        "D_outer":D,"Thickness":th,"t_over_D":td,"R_over_D":rd,
        "file_size_mb":out.stat().st_size/1024**2,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,default=Path("external_workspace/cache_external25"))
    ap.add_argument("--train-stats-dir",type=Path,default=Path("dataeff_workspace/stats/N120"))
    ap.add_argument("--num-time-steps",type=int,default=181)
    ap.add_argument("--world-radius",type=float,default=2.0)
    ap.add_argument("--overwrite",type=int,choices=[0,1],default=0)
    a=ap.parse_args()

    data_dir=a.dataset_dir.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    sources=_discover(data_dir)
    sig=contract_signature(float(a.world_radius),int(a.num_time_steps))
    stats=load_normalization_stats(str(a.train_stats_dir.resolve()))
    if stats["feature_contract"]!=FEATURE_CONTRACT:
        raise RuntimeError(f"Train120 stats feature contract mismatch: {stats['feature_contract']}")
    if stats["shared_cache_signature"]!=sig:
        raise RuntimeError(
            f"Cache signature mismatch: external={sig}, Train120 stats={stats['shared_cache_signature']}"
        )
    if len(stats["train_sample_ids"])!=120:
        raise RuntimeError("External evaluation requires frozen Train120 normalization stats")

    print("="*100)
    print("BUILD EXTERNAL25 INFERENCE CACHE")
    print("dataset     :",data_dir)
    print("output      :",out)
    print("Train stats :",a.train_stats_dir.resolve())
    print("signature   :",sig)
    print("IMPORTANT   : no normalization statistics are fitted from External25")
    print("="*100)

    rows=[]; names={}
    for i,sid in enumerate(sorted(sources),1):
        source=sources[sid]; name=f"external_sample_{sid:04d}.pt"; dest=out/name
        names[sid]=name
        if not a.overwrite and _existing_valid(dest,source,sig,int(a.num_time_steps)):
            p=torch.load(dest,map_location="cpu",weights_only=False)
            r={"sample_id":sid,"nodes":int(p["mesh_pos"].shape[0]),
               "mesh_edges":int(p["mesh_edge_index"].shape[1]),
               "D_outer":float(p["D_outer"]),"Thickness":float(p["Thickness"]),
               "t_over_D":float(p["t_over_D"]),"R_over_D":float(p["R_over_D"]),
               "file_size_mb":dest.stat().st_size/1024**2,"status":"SKIPPED_VALID"}
        else:
            r=_build_one(source,dest,sig,int(a.num_time_steps)); r["status"]="OK"
        rows.append(r)
        print(f"[{i:02d}/25] sample={sid:04d} nodes={r['nodes']} meshE={r['mesh_edges']} size={r['file_size_mb']:.1f} MB {r['status']}",flush=True)

    def write_manifest(filename,ids):
        (out/filename).write_text("\n".join(names[i] for i in ids)+"\n",encoding="utf-8")
    write_manifest("all_manifest.txt",sorted(sources))
    write_manifest("ood15_manifest.txt",OOD15_IDS)
    write_manifest("mesh_variants10_manifest.txt",MESH10_IDS)
    contract={
        "format":"ANARESID_external_eval_contract_v1",
        "feature_contract":FEATURE_CONTRACT,"cache_signature":sig,
        "num_time_steps":int(a.num_time_steps),"world_edge_radius":float(a.world_radius),
        "normalization_source":"frozen Train120 only",
        "train_stats_dir":str(a.train_stats_dir.resolve()),
        "train_sample_ids":stats["train_sample_ids"],
        "external_dataset_dir":str(data_dir),
        "ood15_ids":OOD15_IDS,"ood_labels":OOD_LABELS,
        "mesh_variants":{str(k):{"source_sample_id":v[0],"mesh_size_mm":v[1]} for k,v in MESH_MAP.items()},
        "rows":rows,
    }
    (out/"external_cache_contract.json").write_text(json.dumps(contract,indent=2),encoding="utf-8")
    print("="*100); print("EXTERNAL25 CACHE PASSED"); print("cached=25; OOD15=15; mesh_variants=10"); print("="*100)

if __name__=="__main__": main()
