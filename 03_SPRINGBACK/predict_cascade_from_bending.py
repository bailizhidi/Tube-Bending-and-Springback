#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cascade evaluation: exp04 bending rollout -> frozen AW-SpringMGN -> springback final state.

This evaluator is intentionally inference-only. It does NOT retrain either model.
For every springback test graph (15 geometries x 36 nominal bend angles):
  1) locate the matching exp04 bending rollout frame by sample_id + angle_deg;
  2) extract predicted tube coordinates and align nodes by Abaqus node label;
  3) rebuild AW geometry node/edge features from predicted X_bend;
  4) predict dU_springback with the frozen AW checkpoint;
  5) form X_spring_pred = X_bend_pred + dU_pred;
  6) evaluate final-position error against Abaqus X_spring_true.

IMPORTANT: cascade position error is X_spring_pred - X_spring_true.  It is NOT
just dU_pred - dU_true because X_bend_pred differs from X_bend_true.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from model import MeshGraphNetOneStep
from springback_dataset import GraphData, denormalize_y
from utils import (
    ensure_dir,
    get_angle,
    get_edge_index,
    get_sample_id,
    get_scalar_npz,
    get_target,
    get_x_bend,
    get_x_spring,
    make_edge_features,
)


def load_model(path: str, device: torch.device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    st = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ck["stats"].items()}
    m = MeshGraphNetOneStep(
        int(st["num_node_features"].item()),
        int(st["num_edge_features"].item()),
        int(st["num_output_features"].item()),
        int(cfg.get("hidden_dim", 128)),
        int(cfg.get("num_message_passing_steps", 12)),
        int(cfg.get("mlp_layers", 2)),
        float(cfg.get("dropout", 0.0)),
    ).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, cfg, st, ck


def scalar(z, keys, default=None):
    for key in keys:
        if key in z.files:
            a = np.asarray(z[key])
            if a.size:
                v = a.reshape(-1)[0]
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="ignore")
                try:
                    return v.item()
                except Exception:
                    return v
    return default


def discover_bending_raw(data_dir: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for p in sorted(Path(data_dir).glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            sid = int(float(scalar(z, ["sample_id", "Sample_ID"], -1)))
        if sid in out:
            raise RuntimeError(f"duplicate bending raw Sample_ID={sid}: {out[sid]} and {p}")
        out[sid] = str(p)
    if not out:
        raise RuntimeError(f"no bending raw NPZ files found in {data_dir}")
    return out


def discover_bending_predictions(pred_dir: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    pat = re.compile(r"sample[_-]?0*(\d+).*rollout\.npz$", re.I)
    for p in sorted(Path(pred_dir).glob("*.npz")):
        m = pat.search(p.name)
        if not m:
            continue
        sid = int(m.group(1))
        if sid in out:
            raise RuntimeError(f"duplicate bending rollout Sample_ID={sid}: {out[sid]} and {p}")
        out[sid] = str(p)
    if not out:
        raise RuntimeError(f"no sample_*_rollout.npz files found in {pred_dir}")
    return out


def resolve_split_files(ck: Dict, split: str, fallback_dir: str) -> List[str]:
    if "splits" not in ck or split not in ck["splits"]:
        raise RuntimeError(f"checkpoint does not contain split={split!r}")
    files = []
    for p in ck["splits"][split]:
        p = str(p)
        if os.path.isfile(p):
            files.append(p)
            continue
        q = os.path.join(fallback_dir, os.path.basename(p))
        if os.path.isfile(q):
            files.append(os.path.abspath(q))
            continue
        raise FileNotFoundError(f"springback pair file not found: checkpoint={p}; fallback={q}")
    return files


def build_geom_features(z, path: str, xb_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(xb_pred.shape[0])
    clamp = np.asarray(z["clamp_mask"], dtype=np.float32).reshape(n, 1)
    angle = float(get_angle(z, path))
    angle_norm = angle / 180.0
    t = get_scalar_npz(z, "Thickness", get_scalar_npz(z, "thickness_value", 1.0))
    D = get_scalar_npz(z, "D_outer", 8.0)
    R = get_scalar_npz(z, "R_bending", 24.0)
    scalars = np.repeat(np.asarray([[t, D, R, angle_norm]], dtype=np.float32), n, axis=0)
    x = np.concatenate([xb_pred.astype(np.float32), clamp, scalars], axis=1).astype(np.float32)
    ei = get_edge_index(z, n)
    ea = make_edge_features(xb_pred.astype(np.float32), ei)
    return x, ei, ea


def make_label_order(bend_labels: np.ndarray, spring_labels: np.ndarray) -> np.ndarray:
    bend_labels = np.asarray(bend_labels, dtype=np.int64).reshape(-1)
    spring_labels = np.asarray(spring_labels, dtype=np.int64).reshape(-1)
    if len(set(map(int, bend_labels.tolist()))) != bend_labels.size:
        raise RuntimeError("duplicate labels in bending tube_node_label")
    lut = {int(v): i for i, v in enumerate(bend_labels.tolist())}
    missing = [int(v) for v in spring_labels.tolist() if int(v) not in lut]
    if missing:
        raise RuntimeError(f"springback node labels absent from bending labels; first={missing[:10]}")
    if spring_labels.size != bend_labels.size:
        raise RuntimeError(
            f"tube node count mismatch: springback={spring_labels.size}, bending={bend_labels.size}"
        )
    return np.asarray([lut[int(v)] for v in spring_labels.tolist()], dtype=np.int64)


def metric_triplet(err_vec: np.ndarray) -> Tuple[float, float, float]:
    mag = np.linalg.norm(np.asarray(err_vec, dtype=np.float64), axis=1)
    return float(mag.mean()), float(np.percentile(mag, 95.0)), float(mag.max())


def mean_abs_xyz(err_vec: np.ndarray) -> Tuple[float, float, float]:
    e = np.abs(np.asarray(err_vec, dtype=np.float64))
    return float(e[:, 0].mean()), float(e[:, 1].mean()), float(e[:, 2].mean())


def load_bending_sample(raw_path: str, pred_path: str):
    with np.load(raw_path, allow_pickle=False) as z:
        required = ["angle_deg", "world_pos", "tube_global_index", "tube_node_label"]
        miss = [k for k in required if k not in z.files]
        if miss:
            raise RuntimeError(f"bending raw NPZ missing {miss}: {raw_path}")
        angle_deg = np.asarray(z["angle_deg"], dtype=np.float64).reshape(-1)
        world_pos = np.asarray(z["world_pos"], dtype=np.float32)
        tube_global_index = np.asarray(z["tube_global_index"], dtype=np.int64).reshape(-1)
        tube_node_label = np.asarray(z["tube_node_label"], dtype=np.int64).reshape(-1)
    with np.load(pred_path, allow_pickle=False) as z:
        if "pred_world_pos" not in z.files:
            raise RuntimeError(f"bending rollout missing pred_world_pos: {pred_path}")
        pred_world_pos = np.asarray(z["pred_world_pos"], dtype=np.float32)
        exact_world_pos = np.asarray(z["exact_world_pos"], dtype=np.float32) if "exact_world_pos" in z.files else None

    if world_pos.ndim != 3 or world_pos.shape[2] != 3:
        raise RuntimeError(f"bad bending world_pos shape={world_pos.shape}: {raw_path}")
    if pred_world_pos.shape != world_pos[1:].shape:
        raise RuntimeError(
            f"bending prediction shape mismatch: pred={pred_world_pos.shape}, expected={world_pos[1:].shape}"
        )
    if exact_world_pos is not None:
        if exact_world_pos.shape != world_pos[1:].shape:
            raise RuntimeError(f"bad exact_world_pos shape={exact_world_pos.shape}")
        exact_contract = float(np.max(np.abs(exact_world_pos - world_pos[1:])))
        if exact_contract > 5.0e-5:
            raise RuntimeError(f"bending rollout exact-position contract mismatch={exact_contract:.6e} mm")
    if angle_deg.shape[0] != world_pos.shape[0]:
        raise RuntimeError(f"angle/world frame mismatch: {angle_deg.shape[0]} vs {world_pos.shape[0]}")
    if tube_global_index.size != tube_node_label.size:
        raise RuntimeError("tube_global_index/tube_node_label length mismatch")
    if tube_global_index.min() < 0 or tube_global_index.max() >= world_pos.shape[1]:
        raise RuntimeError("tube_global_index out of range")
    return angle_deg, world_pos, tube_global_index, tube_node_label, pred_world_pos


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["test"])
    ap.add_argument("--springback_data_dir", required=True)
    ap.add_argument("--bending_raw_dir", required=True)
    ap.add_argument("--bending_pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--save_case_npz", action="store_true")
    ap.add_argument("--expected_graphs", type=int, default=540)
    ap.add_argument("--expected_samples", type=int, default=15)
    ap.add_argument("--expected_angles_per_sample", type=int, default=36)
    ap.add_argument("--angle_match_tol_deg", type=float, default=0.26)
    ap.add_argument("--truth_match_tol_mm", type=float, default=5.0e-3)
    ap.add_argument("--preflight_only", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, st, ck = load_model(args.ckpt, device)
    feature_set = str(cfg.get("feature_set", "geom")).strip().lower()
    if feature_set not in {"geom", "a_geom"}:
        raise RuntimeError(
            f"Cascade patch is intentionally for final geometry-only AW checkpoint; got feature_set={feature_set!r}"
        )

    files = resolve_split_files(ck, args.split, args.springback_data_dir)
    if len(files) != args.expected_graphs:
        raise RuntimeError(f"expected {args.expected_graphs} springback test graphs, got {len(files)}")

    grouped: Dict[int, List[str]] = defaultdict(list)
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            sid = int(get_sample_id(z, p))
        grouped[sid].append(p)
    if len(grouped) != args.expected_samples:
        raise RuntimeError(f"expected {args.expected_samples} test samples, got {len(grouped)}: {sorted(grouped)}")
    for sid, ps in grouped.items():
        if len(ps) != args.expected_angles_per_sample:
            raise RuntimeError(
                f"sample {sid:04d}: expected {args.expected_angles_per_sample} angles, got {len(ps)}"
            )

    raw_map = discover_bending_raw(args.bending_raw_dir)
    pred_map = discover_bending_predictions(args.bending_pred_dir)
    miss_raw = [sid for sid in grouped if sid not in raw_map]
    miss_pred = [sid for sid in grouped if sid not in pred_map]
    if miss_raw or miss_pred:
        raise RuntimeError(f"missing bending data: raw={miss_raw}, pred={miss_pred}")

    ensure_dir(args.out_dir)
    case_dir = os.path.join(args.out_dir, "cases_npz")
    if args.save_case_npz:
        ensure_dir(case_dir)

    rows = []
    contract_rows = []
    graph_counter = 0

    node_mean = st["node_mean"].detach().cpu()
    node_std = st["node_std"].detach().cpu()
    edge_mean = st["edge_mean"].detach().cpu()
    edge_std = st["edge_std"].detach().cpu()

    for sid in sorted(grouped):
        angle_arr, bend_world_true, tube_gidx, bend_labels, bend_pred_all = load_bending_sample(
            raw_map[sid], pred_map[sid]
        )
        for path in sorted(grouped[sid]):
            with np.load(path, allow_pickle=False) as z:
                sid2 = int(get_sample_id(z, path))
                ang = float(get_angle(z, path))
                if sid2 != sid:
                    raise RuntimeError(f"group sample mismatch: {sid2} vs {sid}")
                xb_true = get_x_bend(z).astype(np.float32)
                xs_true = get_x_spring(z).astype(np.float32)
                du_true = get_target(z).astype(np.float32)
                spring_labels = np.asarray(z["node_labels"], dtype=np.int64).reshape(-1)
                cells = np.asarray(z["cells"], dtype=np.int64)

                fi = int(np.argmin(np.abs(angle_arr - ang)))
                angle_delta = float(abs(angle_arr[fi] - ang))
                if fi <= 0:
                    raise RuntimeError(f"sample {sid:04d} angle={ang}: matched frame {fi}; rollout has no frame0 prediction")
                if angle_delta > args.angle_match_tol_deg:
                    raise RuntimeError(
                        f"sample {sid:04d} angle={ang}: nearest bending frame angle={angle_arr[fi]} delta={angle_delta}"
                    )
                pi = fi - 1

                order = make_label_order(bend_labels, spring_labels)
                xb_bend_gt = bend_world_true[fi, tube_gidx, :][order]
                xb_pred = bend_pred_all[pi, tube_gidx, :][order].astype(np.float32)

                truth_diff = xb_bend_gt.astype(np.float64) - xb_true.astype(np.float64)
                truth_mean, truth_p95, truth_max = metric_triplet(truth_diff)
                contract_rows.append({
                    "sample_id": sid,
                    "angle_deg": ang,
                    "bending_frame_index": fi,
                    "bending_frame_angle_deg": float(angle_arr[fi]),
                    "truth_bend_mean_mismatch_mm": truth_mean,
                    "truth_bend_p95_mismatch_mm": truth_p95,
                    "truth_bend_max_mismatch_mm": truth_max,
                })
                if truth_max > args.truth_match_tol_mm:
                    raise RuntimeError(
                        f"sample {sid:04d} angle={ang}: bending/springback true X_bend mismatch "
                        f"max={truth_max:.6e} mm > tol={args.truth_match_tol_mm:.6e}"
                    )

                if args.preflight_only:
                    graph_counter += 1
                    continue

                x_raw, ei, ea_raw = build_geom_features(z, path, xb_pred)

            g = GraphData(
                x=(torch.tensor(x_raw) - node_mean) / node_std,
                edge_index=torch.tensor(ei, dtype=torch.long),
                edge_attr=(torch.tensor(ea_raw) - edge_mean) / edge_std,
            ).to(device)
            du_pred = denormalize_y(model(g), st).detach().cpu().numpy().astype(np.float32)
            xs_pred = (xb_pred + du_pred).astype(np.float32)

            bend_err = xb_pred.astype(np.float64) - xb_true.astype(np.float64)
            final_err = xs_pred.astype(np.float64) - xs_true.astype(np.float64)
            du_err = du_pred.astype(np.float64) - du_true.astype(np.float64)
            bend_mean, bend_p95, bend_max = metric_triplet(bend_err)
            final_mean, final_p95, final_max = metric_triplet(final_err)
            du_mean, du_p95, du_max = metric_triplet(du_err)
            fx, fy, fz = mean_abs_xyz(final_err)

            rows.append({
                "file": path,
                "sample_id": sid,
                "angle_deg": ang,
                "bending_frame_index": fi,
                "bending_input_mae_mag": bend_mean,
                "bending_input_p95_mag": bend_p95,
                "bending_input_max_mag": bend_max,
                "cascade_mae_dx": fx,
                "cascade_mae_dy": fy,
                "cascade_mae_dz": fz,
                "cascade_mae_mag": final_mean,
                "cascade_p95_mag": final_p95,
                "cascade_max_mag": final_max,
                "du_model_mae_mag": du_mean,
                "du_model_p95_mag": du_p95,
                "du_model_max_mag": du_max,
                "truth_bend_max_mismatch_mm": truth_max,
            })

            if args.save_case_npz:
                base = "sample_{:04d}_angle_{:03d}".format(sid, int(round(ang)))
                np.savez_compressed(
                    os.path.join(case_dir, base + ".npz"),
                    sample_id=np.asarray([sid], dtype=np.int32),
                    angle_deg=np.asarray([ang], dtype=np.float32),
                    # Keep X_bend as the true bend state so the validated angle evaluator
                    # uses exactly the same end-loop template as Oracle evaluation.
                    X_bend=xb_true.astype(np.float32),
                    X_bend_true=xb_true.astype(np.float32),
                    X_bend_pred=xb_pred.astype(np.float32),
                    X_spring_true=xs_true.astype(np.float32),
                    X_spring_pred=xs_pred.astype(np.float32),
                    dU_true=du_true.astype(np.float32),
                    dU_pred=du_pred.astype(np.float32),
                    cells=cells.astype(np.int32),
                )

            graph_counter += 1
            if graph_counter % 50 == 0 or graph_counter == len(files):
                print(f"[CASCADE] {graph_counter}/{len(files)}", flush=True)

    # Always save the cross-dataset contract audit.
    contract_csv = os.path.join(args.out_dir, "cascade_input_contract.csv")
    with open(contract_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(contract_rows[0].keys()))
        w.writeheader(); w.writerows(contract_rows)
    contract_max = max(float(r["truth_bend_max_mismatch_mm"]) for r in contract_rows)
    print(f"[CONTRACT] cases={len(contract_rows)} max_true_Xbend_mismatch_mm={contract_max:.8e}")

    if args.preflight_only:
        print("CASCADE PREFLIGHT PASSED")
        return

    metrics_csv = os.path.join(args.out_dir, "metrics_cascade_test.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    bend_mean = float(np.mean([r["bending_input_mae_mag"] for r in rows]))
    bend_p95 = float(np.mean([r["bending_input_p95_mag"] for r in rows]))
    bend_max = float(np.max([r["bending_input_max_mag"] for r in rows]))
    final_mean = float(np.mean([r["cascade_mae_mag"] for r in rows]))
    final_p95 = float(np.mean([r["cascade_p95_mag"] for r in rows]))
    final_max = float(np.max([r["cascade_max_mag"] for r in rows]))
    du_mean = float(np.mean([r["du_model_mae_mag"] for r in rows]))
    du_p95 = float(np.mean([r["du_model_p95_mag"] for r in rows]))
    du_max = float(np.max([r["du_model_max_mag"] for r in rows]))

    summary = {
        "graphs": len(rows),
        "test_sample_ids": sorted(int(x) for x in grouped.keys()),
        "bending_input_mean_node_mae_mm": bend_mean,
        "bending_input_mean_graph_p95_mm": bend_p95,
        "bending_input_worst_node_mm": bend_max,
        "cascade_final_mean_node_mae_mm": final_mean,
        "cascade_final_mean_graph_p95_mm": final_p95,
        "cascade_final_worst_node_mm": final_max,
        "springback_increment_model_mean_node_mae_mm": du_mean,
        "springback_increment_model_mean_graph_p95_mm": du_p95,
        "springback_increment_model_worst_node_mm": du_max,
        "max_true_Xbend_cross_dataset_mismatch_mm": contract_max,
        "bending_checkpoint_source": os.path.abspath(args.bending_pred_dir),
        "springback_checkpoint": os.path.abspath(args.ckpt),
    }
    with open(os.path.join(args.out_dir, "cascade_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "[BENDING INPUT] graphs={} mean_node_mae_mm={:.8e} mean_case_p95_mm={:.8e} worst_node_mm={:.8e}".format(
            len(rows), bend_mean, bend_p95, bend_max
        )
    )
    print(
        "[CASCADE FINAL] graphs={} mean_node_mae_mm={:.8e} mean_case_p95_mm={:.8e} worst_node_mm={:.8e}".format(
            len(rows), final_mean, final_p95, final_max
        )
    )
    print(
        "[SPRINGBACK dU MODEL] mean_node_mae_mm={:.8e} mean_case_p95_mm={:.8e} worst_node_mm={:.8e}".format(
            du_mean, du_p95, du_max
        )
    )
    print(metrics_csv)


if __name__ == "__main__":
    main()
