"""
scene_to_glb.py — export a REAL textured mesh GLB of a Holodeck scene.

objathor assets are NOT GLB files: each <uid>/ holds <uid>.pkl.gz
(vertices / triangles / uvs) plus albedo.jpg. This script reads that format
directly, rebuilds each object as a textured trimesh, places it per the THOR
scene metadata, adds a synthesized floor + walls, and exports one GLB.

Run in the same env as export_from_thor.py (conda env `holodeck`). Needs Unity.

Usage:
  python webpage/tools/scene_to_glb.py \
    --house "clean_scene_layout/a_bedroom-2026-04-06-22-09-25-341290/a_bedroom.json" \
    --out   "webpage/models/occ_clean.glb"
"""
import argparse
import gzip
import os
import pickle
import sys

import ai2thor
import ai2thor.wsgi_server
import compress_json
import numpy as np
import trimesh
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from trimesh.transformations import euler_matrix

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from ai2holodeck.constants import THOR_COMMIT_ID, OBJATHOR_ASSETS_DIR


# ── objathor pkl → trimesh ─────────────────────────────────────────────────

def _xyz(lst):
    return np.array([[p["x"], p["y"], p["z"]] for p in lst], dtype=np.float64)


def load_objathor_mesh(asset_dir, uid):
    folder = os.path.join(asset_dir, uid)
    pkl = os.path.join(folder, f"{uid}.pkl.gz")
    if not os.path.isfile(pkl):
        return None
    with gzip.open(pkl, "rb") as f:
        d = pickle.load(f)

    verts = _xyz(d["vertices"])
    faces = np.asarray(d["triangles"], dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # objathor stores a base yaw offset that THOR applies on load.
    yrot = float(d.get("yRotOffset", 0.0) or 0.0)
    if abs(yrot) > 1e-6:
        mesh.apply_transform(euler_matrix(0.0, np.deg2rad(yrot), 0.0, "sxyz"))

    # Texture: albedo.jpg lives next to the pkl (paths inside are docker paths).
    albedo = os.path.join(folder, "albedo.jpg")
    if "uvs" in d and d["uvs"] and os.path.isfile(albedo):
        uv = np.array([[p["x"], p["y"]] for p in d["uvs"]], dtype=np.float64)
        try:
            img = Image.open(albedo).convert("RGB")
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv, image=img,
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=img, metallicFactor=0.0, roughnessFactor=0.9
                ),
            )
        except Exception as e:
            print(f"[WARN] texture failed for {uid}: {e}")
    return mesh


def fit_scale(mesh, aabb_size):
    """Single uniform scale so the mesh roughly matches the THOR AABB."""
    ext = np.asarray(mesh.extents, dtype=np.float64)
    tgt = np.asarray(aabb_size, dtype=np.float64)
    ok = (ext > 1e-9) & (tgt > 1e-9)
    if not ok.any():
        return 1.0
    return float(np.median((tgt[ok] / ext[ok])))


# ── synthesized room shell ─────────────────────────────────────────────────

# Colors sampled from the rendered reference (occ_view_clean.png), brightened
# a little since the render is lit and our viewer relights flat surfaces.
FLOOR_RGBA = (182, 140, 95, 255)    # warm orange-tan tile
WALL_RGBA  = (188, 162, 134, 255)   # warm beige drywall
GLASS_RGBA = (150, 188, 208, 255)   # pale window glass
DOOR_RGBA  = (96, 70, 48, 255)      # dark wood door


def _color(m, rgba):
    m.visual.face_colors = np.tile(np.array(rgba, np.uint8), (len(m.faces), 1))
    return m


def _fan_mesh(verts, color):
    v = np.asarray(verts, dtype=np.float64)
    if len(v) < 3:
        return None
    faces = [[0, i, i + 1] for i in range(1, len(v) - 1)]
    faces = np.array(faces, dtype=np.int64)
    faces = np.vstack([faces, faces[:, ::-1]])
    return _color(trimesh.Trimesh(vertices=v, faces=faces, process=False), color)


def _quad(p0, p1, p2, p3, color):
    v = np.array([p0, p1, p2, p3], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    f = np.vstack([f, f[:, ::-1]])
    return _color(trimesh.Trimesh(vertices=v, faces=f, process=False), color)


def _wall_frame(wall):
    poly = wall.get("polygon") or []
    if len(poly) < 4:
        return None
    P = np.array([[p["x"], p["y"], p["z"]] for p in poly], dtype=np.float64)
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    bottom = P[np.isclose(P[:, 1], ymin)]
    if len(bottom) < 2:
        return None
    b0, b1 = bottom[0], bottom[-1]
    u_vec = b1 - b0
    width = float(np.linalg.norm(u_vec))
    if width < 1e-6:
        return None
    return b0.copy(), u_vec / width, width, float(ymax - ymin)


def _wall_with_holes(wall, openings):
    """Returns (wall_mesh, [(kind, quad_pts), ...]) — wall minus axis-aligned
    rectangular openings, plus a fill quad (window glass / door panel) per hole."""
    fr = _wall_frame(wall)
    if fr is None:
        return None, []
    origin, u_dir, width, height = fr
    up = np.array([0.0, 1.0, 0.0])
    wid = wall.get("id", "")

    holes = []
    for kind, op in openings:
        if wid not in (op.get("wall0"), op.get("wall1")):
            continue
        hp = op.get("holePolygon") or []
        if len(hp) < 2:
            continue
        x0, x1 = sorted([hp[0]["x"], hp[1]["x"]])
        y0, y1 = sorted([hp[0]["y"], hp[1]["y"]])
        if x0 < -0.05 or x1 > width + 0.05:        # x measured from other end
            x0, x1 = width - x1, width - x0
        x0, x1 = max(0.0, x0), min(width, x1)
        y0, y1 = max(0.0, y0), min(height, y1)
        holes.append((kind, x0, x1, y0, y1))

    xs = sorted(set([0.0, width] + [h[1] for h in holes] + [h[2] for h in holes]))
    ys = sorted(set([0.0, height] + [h[3] for h in holes] + [h[4] for h in holes]))
    verts, faces = [], []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(a < cx < b and c < cy < d for _, a, b, c, d in holes):
                continue
            corners = [(xs[i], ys[j]), (xs[i + 1], ys[j]),
                       (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])]
            base = len(verts)
            for (u, v) in corners:
                verts.append(origin + u_dir * u + up * v)
            faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    wall_mesh = None
    if faces:
        f = np.array(faces, dtype=np.int64)
        f = np.vstack([f, f[:, ::-1]])
        wall_mesh = _color(trimesh.Trimesh(vertices=np.array(verts), faces=f,
                                           process=False), WALL_RGBA)

    fills = []
    for kind, a, b, c, d in holes:
        p = lambda u, v: origin + u_dir * u + up * v
        rgba = GLASS_RGBA if kind == "window" else DOOR_RGBA
        fills.append((kind, _quad(p(a, c), p(b, c), p(b, d), p(a, d), rgba)))
    return wall_mesh, fills


def build_shell(house):
    """Returns list of (name, mesh) so the viewer can style by prefix."""
    out = []
    for k, room in enumerate(house.get("rooms", []) or []):
        pts = [(p["x"], p.get("y", 0.0), p["z"]) for p in (room.get("floorPolygon") or [])]
        m = _fan_mesh(pts, FLOOR_RGBA)
        if m:
            out.append((f"floor_{k}", m))

    openings = ([("window", w) for w in (house.get("windows", []) or [])] +
                [("door",   d) for d in (house.get("doors", []) or [])])
    wi = gi = di = 0
    for wall in house.get("walls", []) or []:
        if str(wall.get("id", "")).endswith("exterior"):
            continue
        wm, fills = _wall_with_holes(wall, openings)
        if wm is not None:
            out.append((f"wall_{wi}", wm)); wi += 1
        for kind, fm in fills:
            if kind == "window":
                out.append((f"glass_{gi}", fm)); gi += 1
            else:
                out.append((f"door_{di}", fm)); di += 1
    return out


def _load_cloud(path):
    """Load a colored point cloud (.ply/.npz from export_glb.py) as a
    trimesh.PointCloud (used as the real walls / windows / doors)."""
    if path.endswith(".npz"):
        d = np.load(path)
        xyz = d["xyz"].astype(np.float64)
        rgb = d["rgb"].astype(np.uint8)
    else:
        pc = trimesh.load(path, process=False)
        xyz = np.asarray(pc.vertices, dtype=np.float64)
        rgb = (np.asarray(pc.colors, dtype=np.uint8)[:, :3]
               if pc.colors is not None and len(pc.colors)
               else np.full((len(xyz), 3), 180, np.uint8))
    if len(xyz) == 0:
        return None
    rgba = np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], axis=1)
    return trimesh.PointCloud(vertices=xyz, colors=rgba)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asset-dir", default=OBJATHOR_ASSETS_DIR)
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--obj-offset", default="0,0,0",
                    help="nudge furniture vs the point-cloud room, meters "
                         "'x,y,z' in THOR world coords (before mirroring). "
                         "e.g. '0.05,0,-0.1' to fix small registration drift.")
    ap.add_argument("--mirror", choices=["x", "z", "none"], default="x",
                    help="THOR is left-handed, glTF is right-handed: mirror one "
                         "axis to un-flip left/right. Try 'x'; if it's mirrored "
                         "the other way use 'z'; 'none' to disable.")
    ap.add_argument("--cloud", default=None,
                    help="point cloud .ply/.npz from export_glb.py. If given, "
                         "the room shell comes from the cloud (real walls / "
                         "windows / doors) and the synthesized shell is skipped.")
    ap.add_argument("--strip-furniture-points", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="remove cloud points inside furniture AABBs so the "
                         "cloud is walls/windows/doors only (no double furniture "
                         "with the meshes). --no-strip-furniture-points to keep.")
    ap.add_argument("--strip-pad", type=float, default=1.05,
                    help="enlarge each furniture AABB by this factor when "
                         "stripping cloud points (avoids leaving a point halo).")
    a = ap.parse_args()
    off = np.array([float(v) for v in a.obj_offset.split(",")], dtype=np.float64)

    house = compress_json.load(a.house)

    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        start_unity=True,
        port=a.port,
        scene="Procedural",
        gridSize=0.25,
        width=1024,
        height=1024,
        server_class=ai2thor.wsgi_server.WsgiServer,
        makeAgentsVisible=False,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=a.asset_dir, asset_symlink=True, verbose=False,
        ),
    )
    controller.step(action="CreateHouse", house=house)
    # AABBs / positions aren't populated on the CreateHouse event itself —
    # advance one step so object metadata is filled in.
    event = controller.step("Pass")
    objs = event.metadata.get("objects", []) or []
    n_uid = sum(1 for o in objs if len((o.get("assetId") or "")) == 32)
    n_aabb = sum(1 for o in objs
                 if (o.get("axisAlignedBoundingBox", {}) or {}).get("size"))
    print(f"[OK] CreateHouse — {len(objs)} objects "
          f"({n_uid} with uid assetId, {n_aabb} with AABB)")
    if objs:
        o0 = objs[0]
        print(f"[DBG] sample: assetId={o0.get('assetId')!r} "
              f"aabb={o0.get('axisAlignedBoundingBox', {}).get('size')} "
              f"pos={o0.get('position')}")
    controller.stop()

    scene = trimesh.Scene()
    placed, missing, no_aabb = 0, [], 0
    furn_boxes = []   # (center_xyz, half_xyz) in THOR world coords, pre-mirror
    for obj in objs:
        uid = (obj.get("assetId") or "").lower()
        if len(uid) != 32:
            continue
        mesh = load_objathor_mesh(a.asset_dir, uid)
        if mesh is None:
            missing.append(uid)
            continue

        aabb = obj.get("axisAlignedBoundingBox", {}) or {}
        size = aabb.get("size", {}) or {}
        ctr  = aabb.get("center", {}) or obj.get("position", {}) or {}
        rot  = obj.get("rotation", {}) or {}

        mesh = mesh.copy()
        mesh.apply_translation(-mesh.bounding_box.centroid)

        sz = [size.get("x", 0), size.get("y", 0), size.get("z", 0)]
        if not any(sz):
            no_aabb += 1
        s = fit_scale(mesh, sz)
        S = np.eye(4); S[0, 0] = S[1, 1] = S[2, 2] = s
        R = euler_matrix(np.deg2rad(rot.get("x", 0.0)),
                         np.deg2rad(rot.get("y", 0.0)),
                         np.deg2rad(rot.get("z", 0.0)), "sxyz")
        T = np.eye(4)
        T[0, 3] = ctr.get("x", 0.0) + off[0]
        T[1, 3] = ctr.get("y", 0.0) + off[1]
        T[2, 3] = ctr.get("z", 0.0) + off[2]
        mesh.apply_transform(T @ R @ S)
        scene.add_geometry(mesh, node_name=f"obj_{placed}_{uid[:8]}")
        placed += 1

        if any(sz):
            furn_boxes.append((
                np.array([T[0, 3], T[1, 3], T[2, 3]], dtype=np.float64),
                np.array(sz, dtype=np.float64) * 0.5,
            ))

    if a.cloud:
        cloud = _load_cloud(a.cloud)
        if cloud is None:
            print(f"[ERROR] could not read point cloud: {a.cloud}")
            return
        n0 = len(cloud.vertices)
        if a.strip_furniture_points and furn_boxes:
            v = np.asarray(cloud.vertices)
            inside = np.zeros(len(v), dtype=bool)
            for c, h in furn_boxes:
                hp = h * a.strip_pad
                inside |= np.all(np.abs(v - c) <= hp, axis=1)
            keep = ~inside
            cloud = trimesh.PointCloud(vertices=v[keep],
                                       colors=np.asarray(cloud.colors)[keep])
            print(f"[OK] stripped {n0 - len(cloud.vertices):,} furniture points "
                  f"from cloud ({len(cloud.vertices):,} walls/windows left)")
        scene.add_geometry(cloud, node_name="cloud_0")
        shell_desc = f"point-cloud room ({len(cloud.vertices)} pts)"
    else:
        for name, m in build_shell(house):
            scene.add_geometry(m, node_name=name)
        shell_desc = "synthesized room shell"

    if placed == 0:
        print("[ERROR] No object meshes placed — check asset dir / metadata.")
        return
    if missing:
        print(f"[WARN] {len(missing)} assets missing pkl (skipped): {missing[:10]}")
    if no_aabb:
        print(f"[WARN] {no_aabb} objects had no AABB (scale fell back to 1.0)")

    # Left-handed (THOR) → right-handed (glTF): mirror one axis so left/right
    # matches reality. Negate the axis on every geometry and reverse mesh
    # winding so normals/lighting stay correct.
    if a.mirror != "none":
        ax = 0 if a.mirror == "x" else 2
        for g in scene.geometry.values():
            g.vertices[:, ax] *= -1.0
            if hasattr(g, "faces") and len(getattr(g, "faces", [])):
                g.faces = g.faces[:, ::-1]
                g.fix_normals()
        print(f"[OK] mirrored {a.mirror}-axis (LH→RH)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    scene.export(a.out)
    print(f"[DONE] {a.out} — {placed} objects + {shell_desc}")


if __name__ == "__main__":
    main()
