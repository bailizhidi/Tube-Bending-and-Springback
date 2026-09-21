#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json, os
import numpy as np


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_oracle(pred_dir):
    m = read_csv(os.path.join(pred_dir, "metrics_test.csv"))
    out = {
        "graphs": len(m),
        "mean_node_mae_mm": float(np.mean([float(r["mae_mag"]) for r in m])),
        "mean_graph_p95_mm": float(np.mean([float(r["p95_mag"]) for r in m])),
        "worst_node_mm": float(np.max([float(r["max_mag"]) for r in m])),
    }
    ap = os.path.join(pred_dir, "angle_metrics_centerline.csv")
    if os.path.isfile(ap):
        a = read_csv(ap)
        out["angle_cases"] = len(a)
        out["mean_angle_error_deg"] = float(np.mean([float(r["abs_angle_error_deg"]) for r in a]))
        out["max_angle_error_deg"] = float(np.max([float(r["abs_angle_error_deg"]) for r in a]))
    return out


def aggregate_cascade(pred_dir):
    m = read_csv(os.path.join(pred_dir, "metrics_cascade_test.csv"))
    out = {
        "graphs": len(m),
        "bending_input_mean_node_mae_mm": float(np.mean([float(r["bending_input_mae_mag"]) for r in m])),
        "bending_input_mean_graph_p95_mm": float(np.mean([float(r["bending_input_p95_mag"]) for r in m])),
        "bending_input_worst_node_mm": float(np.max([float(r["bending_input_max_mag"]) for r in m])),
        "mean_node_mae_mm": float(np.mean([float(r["cascade_mae_mag"]) for r in m])),
        "mean_graph_p95_mm": float(np.mean([float(r["cascade_p95_mag"]) for r in m])),
        "worst_node_mm": float(np.max([float(r["cascade_max_mag"]) for r in m])),
        "du_model_mean_node_mae_mm": float(np.mean([float(r["du_model_mae_mag"]) for r in m])),
    }
    ap = os.path.join(pred_dir, "angle_metrics_centerline.csv")
    if os.path.isfile(ap):
        a = read_csv(ap)
        out["angle_cases"] = len(a)
        out["mean_angle_error_deg"] = float(np.mean([float(r["abs_angle_error_deg"]) for r in a]))
        out["max_angle_error_deg"] = float(np.max([float(r["abs_angle_error_deg"]) for r in a]))
    return out


def pct(new, old):
    return 100.0 * (new / old - 1.0) if old != 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade_dir", required=True)
    ap.add_argument("--oracle_dir", required=True)
    args = ap.parse_args()
    o = aggregate_oracle(args.oracle_dir)
    c = aggregate_cascade(args.cascade_dir)
    print("=" * 88)
    print("ORACLE vs CASCADE Test540")
    print("=" * 88)
    print("Oracle  node mean/p95/max (mm): {:.8f} / {:.8f} / {:.8f}".format(
        o["mean_node_mae_mm"], o["mean_graph_p95_mm"], o["worst_node_mm"]))
    print("Cascade node mean/p95/max (mm): {:.8f} / {:.8f} / {:.8f}".format(
        c["mean_node_mae_mm"], c["mean_graph_p95_mm"], c["worst_node_mm"]))
    print("Cascade bending-input mean/p95/max (mm): {:.8f} / {:.8f} / {:.8f}".format(
        c["bending_input_mean_node_mae_mm"], c["bending_input_mean_graph_p95_mm"], c["bending_input_worst_node_mm"]))
    if "mean_angle_error_deg" in o and "mean_angle_error_deg" in c:
        print("Oracle  angle mean/max (deg): {:.8f} / {:.8f}".format(
            o["mean_angle_error_deg"], o["max_angle_error_deg"]))
        print("Cascade angle mean/max (deg): {:.8f} / {:.8f}".format(
            c["mean_angle_error_deg"], c["max_angle_error_deg"]))
        print("Angle mean change vs Oracle : {:+.2f}%".format(pct(c["mean_angle_error_deg"], o["mean_angle_error_deg"])))
    print("Node mean change vs Oracle  : {:+.2f}%".format(pct(c["mean_node_mae_mm"], o["mean_node_mae_mm"])))
    print("Node max change vs Oracle   : {:+.2f}%".format(pct(c["worst_node_mm"], o["worst_node_mm"])))
    out = {"oracle": o, "cascade": c}
    with open(os.path.join(args.cascade_dir, "oracle_vs_cascade_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
