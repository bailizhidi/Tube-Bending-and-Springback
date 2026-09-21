#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centerline-ring angle calculator for springback VTU predictions.

This version is more robust than directly doing PCA on near-end surface nodes:
- It identifies the two open tube-end boundary loops from *_GT_bend.vtu.
- It expands inward by mesh topology and records BFS LEVELS/rings from each end loop.
- For each level, it computes the cross-section centroid.
- The tube-end tangent is fitted from the SEQUENCE OF CENTROIDS, not from all shell surface nodes.

Why this matters:
For large tubes, a near-end shell surface patch can have larger circumferential spread than axial spread,
so PCA of all surface nodes may select a circumferential direction and produce impossible 90-degree angles.
Fitting the line through ring centroids avoids this failure mode.

Expected files in --pred_dir:
    *_GT_bend.vtu
    *_GT_spring.vtu
    *_PRED_spring.vtu

Output:
    angle_metrics_centerline.csv
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque

import numpy as np

VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_POLYGON = 7


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def parse_sample_angle(path):
    name = os.path.basename(path)
    sid = None
    angle = None
    m = re.search(r"sample[_-]?(\d{4})", name, re.IGNORECASE)
    if m:
        sid = int(m.group(1))
    m = re.search(r"angle[_-]?(\d{1,3})", name, re.IGNORECASE)
    if m:
        angle = int(m.group(1))
    else:
        m = re.search(r"(?:^|[_-])(\d{1,3})[_-]?deg", name, re.IGNORECASE)
        if m:
            angle = int(m.group(1))
        else:
            m = re.search(r"Springback[_-]?(\d{1,3})deg", name, re.IGNORECASE)
            if m:
                angle = int(m.group(1))
    return sid, angle


def try_read_with_meshio(path):
    try:
        import meshio  # type: ignore
    except Exception:
        return None
    try:
        mesh = meshio.read(path)
        pts = np.asarray(mesh.points, dtype=np.float64)
        cells = []
        for block in mesh.cells:
            arr = np.asarray(block.data, dtype=np.int64)
            if block.type in ("triangle", "quad", "polygon"):
                cells.extend(arr.tolist())
            elif arr.ndim == 2 and arr.shape[1] >= 3:
                cells.extend(arr.tolist())
        return pts, cells
    except Exception:
        return None


def _strip_xml_namespace(root):
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]


def _parse_data_array_text(data_array, dtype=float):
    text = data_array.text or ""
    text = text.strip()
    if not text:
        return np.array([], dtype=dtype)
    return np.fromstring(text, sep=" ", dtype=dtype)


def read_vtu(path):
    res = try_read_with_meshio(path)
    if res is not None:
        return res

    tree = ET.parse(path)
    root = tree.getroot()
    _strip_xml_namespace(root)

    piece = root.find(".//Piece")
    if piece is None:
        raise RuntimeError(f"VTU 中找不到 Piece: {path}")

    points_node = piece.find("Points")
    if points_node is None:
        raise RuntimeError(f"VTU 中找不到 Points: {path}")
    p_da = points_node.find("DataArray")
    if p_da is None:
        raise RuntimeError(f"VTU 中 Points/DataArray 缺失: {path}")
    if p_da.attrib.get("format", "ascii").lower() not in ("ascii", ""):
        raise RuntimeError(f"内置解析器只支持 ASCII VTU；可安装 meshio 后重试: {path}")
    pts_raw = _parse_data_array_text(p_da, dtype=np.float64)
    pts = pts_raw.reshape((-1, 3))

    cells_node = piece.find("Cells")
    if cells_node is None:
        return pts, []

    arrays = {}
    for da in cells_node.findall("DataArray"):
        name = da.attrib.get("Name", "")
        if da.attrib.get("format", "ascii").lower() not in ("ascii", ""):
            raise RuntimeError(f"内置解析器只支持 ASCII VTU；可安装 meshio 后重试: {path}")
        arrays[name] = _parse_data_array_text(da, dtype=np.int64)

    if "connectivity" not in arrays or "offsets" not in arrays:
        return pts, []

    conn = arrays["connectivity"]
    offsets = arrays["offsets"]
    types = arrays.get("types", np.zeros_like(offsets))

    cells = []
    start = 0
    for i, off in enumerate(offsets):
        end = int(off)
        cell = conn[start:end].astype(np.int64).tolist()
        start = end
        if len(cell) < 3:
            continue
        ctype = int(types[i]) if i < len(types) else None
        if ctype in (VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON, None):
            cells.append(cell)
        elif len(cell) >= 3:
            cells.append(cell)
    return pts, cells


def cell_edges(cells):
    for cell in cells:
        n = len(cell)
        if n < 3:
            continue
        for i in range(n):
            a = int(cell[i])
            b = int(cell[(i + 1) % n])
            if a == b:
                continue
            yield (a, b) if a < b else (b, a)


def build_boundary_components(cells):
    counter = Counter(cell_edges(cells))
    boundary_edges = [e for e, c in counter.items() if c == 1]
    if not boundary_edges:
        return []
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)
    visited = set()
    comps = []
    for s in adj:
        if s in visited:
            continue
        stack = [s]
        visited.add(s)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def build_graph_adj(cells, n_points):
    adj = [[] for _ in range(n_points)]
    seen = set()
    for a, b in cell_edges(cells):
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if 0 <= a < n_points and 0 <= b < n_points:
            adj[a].append(b)
            adj[b].append(a)
    return adj


def select_two_end_loops(points, cells, min_loop_nodes=8):
    comps = build_boundary_components(cells)
    comps = [c for c in comps if len(c) >= min_loop_nodes]
    if len(comps) < 2:
        return None, None, comps

    centers = [points[c].mean(axis=0) for c in comps]
    sizes = np.asarray([len(c) for c in comps], dtype=float)
    max_size = sizes.max()

    # Usually the two open tube-end loops have similar node counts.
    cand = [i for i, s in enumerate(sizes) if s >= 0.5 * max_size]
    if len(cand) < 2:
        cand = list(range(len(comps)))

    # Pick the farthest pair among comparable loops.
    best_pair = None
    best_dist = -1.0
    for ii in range(len(cand)):
        for jj in range(ii + 1, len(cand)):
            i, j = cand[ii], cand[jj]
            d = float(np.linalg.norm(centers[i] - centers[j]))
            if d > best_dist:
                best_dist = d
                best_pair = (i, j)
    if best_pair is None:
        return None, None, comps
    return comps[best_pair[0]], comps[best_pair[1]], comps


def bfs_levels_from_loop(loop_nodes, adj, rings):
    """Return list of node lists at exact topological distance 0..rings from the boundary loop."""
    start = [int(i) for i in loop_nodes]
    visited = set(start)
    q = deque((i, 0) for i in start)
    levels = [[] for _ in range(rings + 1)]
    for i in start:
        levels[0].append(i)

    while q:
        u, d = q.popleft()
        if d >= rings:
            continue
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                nd = d + 1
                if nd <= rings:
                    levels[nd].append(v)
                q.append((v, nd))
    return [sorted(x) for x in levels]


def fit_axis_from_centroids(points, levels, min_level_nodes=4, min_centroids=3):
    """Fit tangent from cross-section centroid sequence, not surface node cloud."""
    centroids = []
    level_ids = []
    for k, nodes in enumerate(levels):
        if len(nodes) >= min_level_nodes:
            c = points[np.asarray(nodes, dtype=np.int64)].mean(axis=0)
            centroids.append(c)
            level_ids.append(k)
    if len(centroids) < min_centroids:
        raise RuntimeError(f"有效中心点数量不足: {len(centroids)} < {min_centroids}")

    C = np.asarray(centroids, dtype=np.float64)

    # Prefer endpoint-to-interior direction if enough levels are clean.
    # Use PCA of centroids for noise robustness, then orient inward.
    C0 = C - C.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(C0, full_matrices=False)
    axis = vh[0]
    n = np.linalg.norm(axis)
    if n < 1e-12:
        raise RuntimeError("中心线 PCA 主方向长度接近 0")
    axis = axis / n

    inward = C[-1] - C[0]
    if np.linalg.norm(inward) > 1e-12 and float(np.dot(axis, inward)) < 0:
        axis = -axis

    return axis, C, level_ids


def choose_angle(axis1, axis2, nominal_angle=None):
    """
    Compute physical bending angle from two inward-oriented end tangents.

    fit_axis_from_centroids() already orients each end tangent from the
    tube end toward the tube interior. Therefore an initially straight
    tube gives an inter-tangent angle of 180 deg, corresponding to a
    physical bending angle of 0 deg.

    This deterministic definition avoids the acute/obtuse branch
    ambiguity around nominal_angle = 90 deg.
    """
    dot = float(np.clip(np.dot(axis1, axis2), -1.0, 1.0))

    inward_angle = math.degrees(math.acos(dot))
    bend_angle = 180.0 - inward_angle

    acute = math.degrees(math.acos(abs(dot)))
    obtuse = 180.0 - acute

    return (
        float(bend_angle),
        float(inward_angle),
        float(acute),
        float(obtuse),
        "inward_tangent",
        dot,
    )


def make_template(ref_path, rings=20, min_loop_nodes=8):
    points, cells = read_vtu(ref_path)
    if len(cells) == 0:
        raise RuntimeError(f"无法从模板 VTU 读取面单元: {ref_path}")
    loop1, loop2, comps = select_two_end_loops(points, cells, min_loop_nodes=min_loop_nodes)
    if loop1 is None or loop2 is None:
        raise RuntimeError(f"无法识别两个端口 loop: {ref_path}; boundary_components={len(comps)}")
    adj = build_graph_adj(cells, len(points))
    levels1 = bfs_levels_from_loop(loop1, adj, rings=rings)
    levels2 = bfs_levels_from_loop(loop2, adj, rings=rings)
    return {
        "points_ref": points,
        "cells": cells,
        "loop1": loop1,
        "loop2": loop2,
        "levels1": levels1,
        "levels2": levels2,
        "n_boundary_components": len(comps),
        "loop1_nodes": len(loop1),
        "loop2_nodes": len(loop2),
        "rings": rings,
        "template_file": ref_path,
    }


def compute_angle_with_template(points, template, nominal_angle=None, min_level_nodes=4, min_centroids=3):
    ax1, C1, ids1 = fit_axis_from_centroids(points, template["levels1"], min_level_nodes=min_level_nodes, min_centroids=min_centroids)
    ax2, C2, ids2 = fit_axis_from_centroids(points, template["levels2"], min_level_nodes=min_level_nodes, min_centroids=min_centroids)
    ang, signed, acute, obtuse, mode, dot = choose_angle(ax1, ax2, nominal_angle)
    return {
        "angle_deg": ang,
        "angle_signed_deg": signed,
        "angle_acute_deg": acute,
        "angle_obtuse_deg": obtuse,
        "mode": mode,
        "dot": dot,
        "axis1_x": float(ax1[0]), "axis1_y": float(ax1[1]), "axis1_z": float(ax1[2]),
        "axis2_x": float(ax2[0]), "axis2_y": float(ax2[1]), "axis2_z": float(ax2[2]),
        "center1_start_x": float(C1[0, 0]), "center1_start_y": float(C1[0, 1]), "center1_start_z": float(C1[0, 2]),
        "center1_end_x": float(C1[-1, 0]), "center1_end_y": float(C1[-1, 1]), "center1_end_z": float(C1[-1, 2]),
        "center2_start_x": float(C2[0, 0]), "center2_start_y": float(C2[0, 1]), "center2_start_z": float(C2[0, 2]),
        "center2_end_x": float(C2[-1, 0]), "center2_end_y": float(C2[-1, 1]), "center2_end_z": float(C2[-1, 2]),
        "n_centroids1": len(C1),
        "n_centroids2": len(C2),
        "level_ids1": ";".join(map(str, ids1)),
        "level_ids2": ";".join(map(str, ids2)),
    }


def collect_triplets(pred_dir):
    gt_spring_files = sorted(glob.glob(os.path.join(pred_dir, "**", "*_GT_spring.vtu"), recursive=True), key=natural_key)
    triplets = []
    missing = []
    for gt_spring in gt_spring_files:
        pred_spring = gt_spring.replace("_GT_spring.vtu", "_PRED_spring.vtu")
        gt_bend = gt_spring.replace("_GT_spring.vtu", "_GT_bend.vtu")
        if not os.path.exists(pred_spring):
            missing.append((gt_spring, pred_spring, "missing PRED_spring"))
            continue
        if not os.path.exists(gt_bend):
            gt_bend = gt_spring
        triplets.append((gt_bend, gt_spring, pred_spring))
    return triplets, missing


def main():
    ap = argparse.ArgumentParser(description="Centerline-ring tube angle calculator for springback predictions.")
    ap.add_argument("--pred_dir", required=True, help="predict.py 输出目录")
    ap.add_argument("--out_csv", default=None, help="输出 CSV，默认 pred_dir/angle_metrics_centerline.csv")
    ap.add_argument("--rings", type=int, default=20, help="从端部 loop 沿网格拓扑向内扩展的环数，默认 20")
    ap.add_argument("--min_loop_nodes", type=int, default=8, help="候选端口 loop 的最少节点数")
    ap.add_argument("--min_level_nodes", type=int, default=4, help="一个拓扑层至少多少节点才用于中心点拟合")
    ap.add_argument("--min_centroids", type=int, default=3, help="拟合中心线至少需要多少个中心点")
    ap.add_argument("--no_nominal_disambiguation", action="store_true", help="不使用名义角做 acute/obtuse 消歧")
    args = ap.parse_args()

    out_csv = args.out_csv or os.path.join(args.pred_dir, "angle_metrics_centerline.csv")
    triplets, missing = collect_triplets(args.pred_dir)
    if not triplets:
        print(f"ERROR: 未找到可配对的 VTU: {args.pred_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    failures = []
    print(f"发现 {len(triplets)} 组三文件: GT_bend/GT_spring/PRED_spring")
    print(f"方法: boundary loop -> 拓扑层中心点 -> 中心线切线, rings={args.rings}")

    for idx, (gt_bend, gt_spring, pred_spring) in enumerate(triplets, 1):
        sid, nominal = parse_sample_angle(gt_spring)
        nominal_for_calc = None if args.no_nominal_disambiguation else nominal
        try:
            template = make_template(gt_bend, rings=args.rings, min_loop_nodes=args.min_loop_nodes)
            pts_true, _ = read_vtu(gt_spring)
            pts_pred, _ = read_vtu(pred_spring)
            if pts_true.shape != pts_pred.shape:
                raise RuntimeError(f"GT/PRED 点数不一致: {pts_true.shape} vs {pts_pred.shape}")
            if pts_true.shape[0] != len(template["points_ref"]):
                raise RuntimeError(f"模板与 GT_spring 点数不一致: {len(template['points_ref'])} vs {pts_true.shape[0]}")

            true_info = compute_angle_with_template(
                pts_true, template, nominal_for_calc,
                min_level_nodes=args.min_level_nodes,
                min_centroids=args.min_centroids,
            )
            pred_info = compute_angle_with_template(
                pts_pred, template, nominal_for_calc,
                min_level_nodes=args.min_level_nodes,
                min_centroids=args.min_centroids,
            )

            true_angle = true_info["angle_deg"]
            pred_angle = pred_info["angle_deg"]
            err = pred_angle - true_angle

            if nominal is not None:
                true_springback = float(nominal) - true_angle
                pred_springback = float(nominal) - pred_angle
                springback_err = pred_springback - true_springback
                true_nominal_diff = true_angle - float(nominal)
            else:
                true_springback = np.nan
                pred_springback = np.nan
                springback_err = np.nan
                true_nominal_diff = np.nan

            row = {
                "sample_id": sid if sid is not None else "",
                "nominal_angle_deg": nominal if nominal is not None else "",
                "true_angle_deg": true_angle,
                "pred_angle_deg": pred_angle,
                "angle_error_deg": err,
                "abs_angle_error_deg": abs(err),
                "true_springback_deg": true_springback,
                "pred_springback_deg": pred_springback,
                "springback_error_deg": springback_err,
                "abs_springback_error_deg": abs(springback_err) if not np.isnan(springback_err) else np.nan,
                "true_minus_nominal_deg": true_nominal_diff,
                "gt_mode": true_info["mode"],
                "pred_mode": pred_info["mode"],
                "boundary_components": template["n_boundary_components"],
                "loop1_nodes": template["loop1_nodes"],
                "loop2_nodes": template["loop2_nodes"],
                "n_centroids1": true_info["n_centroids1"],
                "n_centroids2": true_info["n_centroids2"],
                "level_ids1": true_info["level_ids1"],
                "level_ids2": true_info["level_ids2"],
                "template_file": os.path.relpath(gt_bend, args.pred_dir),
                "gt_file": os.path.relpath(gt_spring, args.pred_dir),
                "pred_file": os.path.relpath(pred_spring, args.pred_dir),
            }
            rows.append(row)
            print(f"[{idx}/{len(triplets)}] sample={row['sample_id']} nominal={row['nominal_angle_deg']} true={true_angle:.4f} pred={pred_angle:.4f} err={err:+.4f}")
        except Exception as e:
            failures.append((gt_bend, gt_spring, pred_spring, str(e)))
            print(f"[{idx}/{len(triplets)}] FAILED: {gt_spring} | {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "sample_id", "nominal_angle_deg",
        "true_angle_deg", "pred_angle_deg", "angle_error_deg", "abs_angle_error_deg",
        "true_springback_deg", "pred_springback_deg", "springback_error_deg", "abs_springback_error_deg",
        "true_minus_nominal_deg", "gt_mode", "pred_mode",
        "boundary_components", "loop1_nodes", "loop2_nodes", "n_centroids1", "n_centroids2", "level_ids1", "level_ids2",
        "template_file", "gt_file", "pred_file",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        fail_csv = os.path.splitext(out_csv)[0] + "_failures.csv"
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["template_file", "gt_file", "pred_file", "error"])
            w.writerows(failures)
        print(f"有 {len(failures)} 个样本失败，详见: {fail_csv}")

    if missing:
        miss_csv = os.path.splitext(out_csv)[0] + "_missing_pairs.csv"
        with open(miss_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gt_file", "expected_file", "reason"])
            w.writerows(missing)
        print(f"有 {len(missing)} 个文件缺少配对，详见: {miss_csv}")

    print(f"\n角度结果已保存: {out_csv}")

    if rows:
        errs = np.asarray([r["abs_angle_error_deg"] for r in rows], dtype=float)
        signed = np.asarray([r["angle_error_deg"] for r in rows], dtype=float)
        print("\n========== 中心线端部法角度误差统计 ==========")
        print(f"样本数              : {len(rows)}")
        print(f"Mean |angle error| : {errs.mean():.6f} deg")
        print(f"Median |error|     : {np.median(errs):.6f} deg")
        print(f"Max |error|        : {errs.max():.6f} deg")
        print(f"Mean signed error  : {signed.mean():.6f} deg")

        by_sample = defaultdict(list)
        for r in rows:
            by_sample[r["sample_id"]].append(r["abs_angle_error_deg"])
        print("\n按 sample 的 Mean |angle error|:")
        for sid in sorted(by_sample):
            arr = np.asarray(by_sample[sid], dtype=float)
            print(f"  sample {sid}: mean={arr.mean():.6f}, max={arr.max():.6f}, n={len(arr)}")


if __name__ == "__main__":
    main()
