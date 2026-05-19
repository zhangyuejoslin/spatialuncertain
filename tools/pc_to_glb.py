"""
pc_to_glb.py — convert a colored point cloud (.ply / .npz from export_glb.py)
into a GLB point cloud that <model-viewer> can display with auto-rotate.

Run inside the SAME python env you use for export_glb.py (needs numpy + trimesh).

Usage:
    python webpage/tools/pc_to_glb.py input.ply  webpage/models/clean.glb
    python webpage/tools/pc_to_glb.py input.npz  webpage/models/occluder.glb

Optional flags:
    --voxel 0.05     downsample to one point per 5 cm voxel (smaller GLB)
    --max-points 300000   hard cap on point count (random subsample)
"""
import argparse
import os
import sys

import numpy as np
import trimesh


def load_points(path):
    if path.endswith(".npz"):
        d = np.load(path)
        return d["xyz"].astype(np.float32), d["rgb"].astype(np.uint8)
    if path.endswith(".ply"):
        pc = trimesh.load(path, process=False)
        xyz = np.asarray(pc.vertices, dtype=np.float32)
        if pc.colors is not None and len(pc.colors):
            rgb = np.asarray(pc.colors, dtype=np.uint8)[:, :3]
        else:
            rgb = np.full((len(xyz), 3), 180, np.uint8)
        return xyz, rgb
    sys.exit(f"Unsupported input: {path} (use .ply or .npz)")


def voxel_downsample(xyz, rgb, voxel):
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], rgb[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--voxel", type=float, default=0.0)
    ap.add_argument("--max-points", type=int, default=0)
    a = ap.parse_args()

    xyz, rgb = load_points(a.input)
    print(f"[load] {len(xyz):,} points from {a.input}")

    if a.voxel > 0:
        xyz, rgb = voxel_downsample(xyz, rgb, a.voxel)
        print(f"[voxel {a.voxel}m] -> {len(xyz):,} points")

    if a.max_points and len(xyz) > a.max_points:
        sel = np.random.default_rng(42).choice(len(xyz), a.max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
        print(f"[cap] -> {len(xyz):,} points")

    # Recenter on origin and put the floor near y=0 so model-viewer frames it well.
    center = xyz.mean(axis=0)
    xyz = xyz - center

    rgba = np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], axis=1)
    cloud = trimesh.PointCloud(vertices=xyz, colors=rgba)

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    cloud.export(a.output)  # trimesh writes a glTF POINTS primitive
    size_mb = os.path.getsize(a.output) / 1e6
    print(f"[ok] wrote {a.output}  ({size_mb:.1f} MB, {len(xyz):,} points)")


if __name__ == "__main__":
    main()
