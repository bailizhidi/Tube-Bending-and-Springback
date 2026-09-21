from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from graph_builder import build_world_edge_index, relative_edge4
from packed_dataset import load_normalization_stats, safe_torch_load

# source geometry -> (fine 0.8 mm external, coarse 1.25 mm external)
MESH_GROUPS = {
    4: (183, 184),
    3: (185, 186),
    139: (174, 175),
    15: (187, 188),
    128: (178, 179),
}


def q(x: torch.Tensor, p: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float(), p).item())


def stats_1d(x: torch.Tensor, prefix: str) -> Dict[str, float]:
    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_p05": q(x, 0.05),
        f"{prefix}_p50": q(x, 0.50),
        f"{prefix}_p95": q(x, 0.95),
        f"{prefix}_max": float(x.max().item()),
    }


def degree_stats(edge_index: torch.Tensor, n: int, mask: torch.Tensor | None, prefix: str) -> Dict[str, float]:
    if edge_index.numel() == 0:
        deg = torch.zeros(n, dtype=torch.float32, device=edge_index.device)
    else:
        deg = torch.bincount(edge_index[0].long(), minlength=n).float()
    if mask is not None:
        deg = deg[mask]
    return stats_1d(deg, prefix)


def normalize_edge4(raw4: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(raw4.device, raw4.dtype).reshape(1, 4)
    std = std.to(raw4.device, raw4.dtype).reshape(1, 4)
    std = torch.where(std.abs() < 1e-8, torch.ones_like(std), std)
    return (raw4 - mean) / std


def edge_distribution(raw4: torch.Tensor, edge_mean: torch.Tensor, edge_std: torch.Tensor,
                      edge_min: torch.Tensor, edge_max: torch.Tensor, prefix: str) -> Dict[str, float]:
    if raw4.numel() == 0:
        out = stats_1d(torch.empty(0, device=raw4.device), f"{prefix}_len_mm")
        out.update({
            f"{prefix}_zlen_abs_p95": float("nan"),
            f"{prefix}_zlen_abs_max": float("nan"),
            f"{prefix}_z_any_abs_gt3_frac": float("nan"),
            f"{prefix}_raw_outside_train_range_frac": float("nan"),
            f"{prefix}_len_outside_train_range_frac": float("nan"),
        })
        return out

    length = raw4[:, 3].float()
    z = normalize_edge4(raw4.float(), edge_mean, edge_std)
    zabs = z.abs()
    emin = edge_min.to(raw4.device, raw4.dtype).reshape(1, 4)
    emax = edge_max.to(raw4.device, raw4.dtype).reshape(1, 4)
    outside = (raw4 < emin) | (raw4 > emax)
    len_outside = outside[:, 3]

    out = stats_1d(length, f"{prefix}_len_mm")
    out.update({
        f"{prefix}_zlen_abs_p95": q(zabs[:, 3], 0.95),
        f"{prefix}_zlen_abs_max": float(zabs[:, 3].max().item()),
        f"{prefix}_z_any_abs_gt3_frac": float((zabs > 3.0).any(dim=1).float().mean().item()),
        f"{prefix}_raw_outside_train_range_frac": float(outside.any(dim=1).float().mean().item()),
        f"{prefix}_len_outside_train_range_frac": float(len_outside.float().mean().item()),
    })
    return out


def load_train_map(cache_dir: Path) -> Dict[int, Path]:
    manifest = cache_dir / "train_manifest.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    out = {}
    for name in [x.strip() for x in manifest.read_text().splitlines() if x.strip()]:
        p = cache_dir / name
        packed = safe_torch_load(str(p))
        out[int(packed["sample_id"])] = p
    return out


def load_external_map(cache_dir: Path) -> Dict[int, Path]:
    out = {}
    for p in sorted(cache_dir.glob("external_sample_*.pt")):
        packed = safe_torch_load(str(p))
        out[int(packed["sample_id"])] = p
    return out


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def static_row(packed: Dict, *, source_id: int, eval_id: int, h: float,
               edge_mean, edge_std, edge_min, edge_max, device) -> Dict:
    mesh = packed["mesh_pos"].to(device).float()
    tube = packed["is_tube_node"].to(device).reshape(-1).bool()
    mesh_idx = packed["mesh_edge_index"].to(device).long()
    raw = packed["mesh_edge_attr_static_raw"].to(device).float()
    n = int(mesh.shape[0])
    nt = int(tube.sum().item())
    row = {
        "source_sample_id": source_id,
        "eval_sample_id": eval_id,
        "mesh_size_mm": h,
        "num_nodes": n,
        "num_tube_nodes": nt,
        "tube_node_fraction": nt / max(n, 1),
        "num_mesh_edges": int(mesh_idx.shape[1]),
        "mesh_edges_per_node": float(mesh_idx.shape[1]) / max(n, 1),
    }
    row.update(degree_stats(mesh_idx, n, None, "mesh_degree_all"))
    row.update(degree_stats(mesh_idx, n, tube, "mesh_degree_tube"))
    row.update(edge_distribution(raw, edge_mean, edge_std, edge_min, edge_max, "mesh"))
    return row


def world_rows(packed: Dict, *, source_id: int, eval_id: int, h: float, frames: Iterable[int],
               radius: float, edge_mean, edge_std, edge_min, edge_max, device) -> List[Dict]:
    mesh_idx = packed["mesh_edge_index"].to(device).long().contiguous()
    tube = packed["is_tube_node"].to(device).reshape(-1).bool()
    world_all = packed["world_pos"]
    n = int(packed["mesh_pos"].shape[0])
    rows=[]
    for t in frames:
        world = world_all[t].to(device).float().contiguous()
        idx = build_world_edge_index(world, mesh_idx, tube, radius=float(radius))
        raw = relative_edge4(world, idx) if idx.numel() else torch.empty((0,4),device=device)
        r = {
            "source_sample_id": source_id,
            "eval_sample_id": eval_id,
            "mesh_size_mm": h,
            "frame": int(t),
            "world_radius_mm": float(radius),
            "num_world_edges": int(idx.shape[1]),
            "world_edges_per_node": float(idx.shape[1]) / max(n, 1),
        }
        r.update(degree_stats(idx, n, None, "world_degree_all"))
        r.update(degree_stats(idx, n, tube, "world_degree_tube"))
        r.update(edge_distribution(raw, edge_mean, edge_std, edge_min, edge_max, "world"))
        rows.append(r)
        del world, idx, raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def mean_of(rows: List[Dict], key: str) -> float:
    vals=[float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def paired_rows(static_rows: List[Dict], world_rows_all: List[Dict]) -> List[Dict]:
    s_by={(int(r["source_sample_id"]),float(r["mesh_size_mm"])):r for r in static_rows}
    out=[]
    for src in MESH_GROUPS:
        b=s_by[(src,1.0)]
        wb=[r for r in world_rows_all if int(r["source_sample_id"])==src and float(r["mesh_size_mm"])==1.0]
        for h in (0.8,1.25):
            x=s_by[(src,h)]
            wx=[r for r in world_rows_all if int(r["source_sample_id"])==src and float(r["mesh_size_mm"])==h]
            row={"source_sample_id":src,"mesh_size_mm":h}
            metrics=[
                "num_nodes","num_mesh_edges","mesh_degree_tube_mean","mesh_len_mm_mean",
                "mesh_zlen_abs_p95","mesh_raw_outside_train_range_frac",
            ]
            for k in metrics:
                bv=float(b[k]); xv=float(x[k])
                row[f"{k}_ratio_vs_1mm"]=xv/max(abs(bv),1e-30)
                row[f"{k}_delta_vs_1mm"]=xv-bv
            wmetrics=[
                "num_world_edges","world_edges_per_node","world_degree_tube_mean",
                "world_len_mm_mean","world_zlen_abs_p95","world_raw_outside_train_range_frac",
            ]
            for k in wmetrics:
                bv=mean_of(wb,k); xv=mean_of(wx,k)
                row[f"{k}_mean_ratio_vs_1mm"]=xv/max(abs(bv),1e-30)
                row[f"{k}_mean_delta_vs_1mm"]=xv-bv
            out.append(row)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--external-cache-dir",type=Path,default=Path("external_workspace/cache_external25"))
    ap.add_argument("--baseline-cache-dir",type=Path,default=Path("dataeff_workspace/cache_views/N120"))
    ap.add_argument("--train-stats-dir",type=Path,default=Path("dataeff_workspace/stats/N120"))
    ap.add_argument("--frames",default="0,45,90,135,179")
    ap.add_argument("--world-radius",type=float,default=2.0)
    ap.add_argument("--out-dir",type=Path,default=Path("external_results/mesh_graph_diagnosis"))
    a=ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because PhysicsNeMo radius_search is used")
    device=torch.device("cuda:0")
    frames=[int(x) for x in a.frames.split(",") if x.strip()]
    if any(t<0 or t>179 for t in frames):
        raise ValueError(f"frames must lie in 0..179, got {frames}")

    norm=load_normalization_stats(str(a.train_stats_dir.resolve()))
    raw=norm["raw"]
    edge_mean=norm["edge_mean"].to(device)
    edge_std=norm["edge_std"].to(device)
    edge_min=torch.tensor(raw["edge_min"],dtype=torch.float32,device=device)
    edge_max=torch.tensor(raw["edge_max"],dtype=torch.float32,device=device)

    train=load_train_map(a.baseline_cache_dir.resolve())
    ext=load_external_map(a.external_cache_dir.resolve())

    static=[]; world=[]
    cases=[]
    for src,(fine,coarse) in MESH_GROUPS.items():
        cases += [(src,src,1.0,train[src]),(src,fine,0.8,ext[fine]),(src,coarse,1.25,ext[coarse])]

    print("="*110)
    print("MESH GRAPH STATISTICS DIAGNOSIS")
    print("cases       :",len(cases))
    print("frames      :",frames)
    print("world radius:",a.world_radius,"mm")
    print("edge stats  : frozen Train120 only")
    print("IMPORTANT   : GT world positions are used only for graph-structure diagnosis; no model inference occurs")
    print("="*110)

    for i,(src,eid,h,path) in enumerate(cases,1):
        p=safe_torch_load(str(path))
        sr=static_row(p,source_id=src,eval_id=eid,h=h,edge_mean=edge_mean,edge_std=edge_std,
                      edge_min=edge_min,edge_max=edge_max,device=device)
        static.append(sr)
        wr=world_rows(p,source_id=src,eval_id=eid,h=h,frames=frames,radius=a.world_radius,
                      edge_mean=edge_mean,edge_std=edge_std,edge_min=edge_min,edge_max=edge_max,device=device)
        world.extend(wr)
        print(
            f"[{i:02d}/15] src={src:03d} eval={eid:04d} h={h:.2f} "
            f"N={sr['num_nodes']} meshE={sr['num_mesh_edges']} "
            f"meshDegTube={sr['mesh_degree_tube_mean']:.2f} "
            f"meshLen={sr['mesh_len_mm_mean']:.4f} "
            f"worldE(mean)={mean_of(wr,'num_world_edges'):.1f} "
            f"worldDegTube(mean)={mean_of(wr,'world_degree_tube_mean'):.2f}",
            flush=True,
        )
        del p
        torch.cuda.empty_cache()

    paired=paired_rows(static,world)
    out=a.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"static_mesh_stats.csv",static)
    write_csv(out/"world_graph_stats_by_frame.csv",world)
    write_csv(out/"paired_vs_1mm.csv",paired)

    # concise aggregate by mesh size
    agg=[]
    for h in (0.8,1.0,1.25):
        ss=[r for r in static if float(r["mesh_size_mm"])==h]
        ww=[r for r in world if float(r["mesh_size_mm"])==h]
        agg.append({
            "mesh_size_mm":h,
            "num_cases":len(ss),
            "nodes_mean":float(np.mean([r["num_nodes"] for r in ss])),
            "mesh_edges_per_node_mean":float(np.mean([r["mesh_edges_per_node"] for r in ss])),
            "mesh_degree_tube_mean":float(np.mean([r["mesh_degree_tube_mean"] for r in ss])),
            "mesh_len_mm_mean":float(np.mean([r["mesh_len_mm_mean"] for r in ss])),
            "mesh_zlen_abs_p95_mean":float(np.mean([r["mesh_zlen_abs_p95"] for r in ss])),
            "mesh_outside_train_range_frac_mean":float(np.mean([r["mesh_raw_outside_train_range_frac"] for r in ss])),
            "world_edges_per_node_mean":float(np.mean([r["world_edges_per_node"] for r in ww])),
            "world_degree_tube_mean":float(np.mean([r["world_degree_tube_mean"] for r in ww])),
            "world_len_mm_mean":float(np.mean([r["world_len_mm_mean"] for r in ww])),
            "world_zlen_abs_p95_mean":float(np.mean([r["world_zlen_abs_p95"] for r in ww])),
            "world_outside_train_range_frac_mean":float(np.mean([r["world_raw_outside_train_range_frac"] for r in ww])),
        })
    write_csv(out/"aggregate_by_mesh_size.csv",agg)

    summary={
        "format":"ANARESID_mesh_graph_diagnosis_v1",
        "frames":frames,
        "world_radius_mm":a.world_radius,
        "normalization_source":"frozen Train120 only",
        "aggregate_by_mesh_size":agg,
        "interpretation_rules":{
            "world_degree_shift":"If 0.8/1.25 world_degree_tube_mean differs strongly from 1.0, fixed-radius neighborhood density is mesh-sensitive.",
            "mesh_edge_scale_shift":"If mesh_len_mm or normalized z-length changes strongly, mesh-edge scale is out of the training distribution.",
            "normalization_shift":"High outside_train_range_frac or |z| indicates edge normalization OOD.",
        },
    }
    (out/"diagnosis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    print("\nAGGREGATE BY MESH SIZE")
    for r in agg:
        print(json.dumps(r,ensure_ascii=False))
    print("\nWROTE:",out)
    print("  static_mesh_stats.csv")
    print("  world_graph_stats_by_frame.csv")
    print("  paired_vs_1mm.csv")
    print("  aggregate_by_mesh_size.csv")
    print("  diagnosis_summary.json")

if __name__=="__main__":
    main()
