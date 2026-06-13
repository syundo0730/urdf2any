"""Mesh loading via trimesh, with per-file caching.

Used by the USD backend (which authors mesh geometry inline). The MJCF
backend references mesh files natively and only needs format conversion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

MUJOCO_NATIVE_FORMATS = {".stl", ".obj", ".msh", ".ply"}


@dataclass
class SubMesh:
    """A single renderable mesh with one optional material color."""

    name: str
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3)
    normals: np.ndarray | None  # (N, 3) vertex normals
    uv: np.ndarray | None  # (N, 2)
    color: np.ndarray | None  # (4,) rgba 0-1


def _color_of(geom: trimesh.Trimesh) -> np.ndarray | None:
    visual = getattr(geom, "visual", None)
    if visual is None:
        return None
    material = getattr(visual, "material", None)
    if material is not None:
        for attr in ("baseColorFactor", "diffuse"):
            value = getattr(material, attr, None)
            if value is not None:
                rgba = np.asarray(value, dtype=float)
                if rgba.max() > 1.0:
                    rgba = rgba / 255.0
                return rgba
    return None


@cache
def load_submeshes(path: Path) -> tuple[SubMesh, ...]:
    """Load a mesh file into one or more SubMesh (one per material group)."""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        items = [(name, g) for name, g in loaded.geometry.items() if isinstance(g, trimesh.Trimesh)]
    else:
        items = [(path.stem, loaded)]

    out = []
    for name, geom in items:
        uv = None
        visual = getattr(geom, "visual", None)
        if visual is not None and getattr(visual, "uv", None) is not None and len(visual.uv):
            uv = np.asarray(visual.uv, dtype=float)
        out.append(
            SubMesh(
                name=name,
                vertices=np.asarray(geom.vertices, dtype=float),
                faces=np.asarray(geom.faces, dtype=int),
                normals=np.asarray(geom.vertex_normals, dtype=float),
                uv=uv,
                color=_color_of(geom),
            )
        )
    if not out:
        raise ValueError(f"No mesh geometry found in {path}")
    return tuple(out)


def ensure_mujoco_compatible(path: Path, out_dir: Path) -> Path:
    """Copy (or convert) a mesh file into ``out_dir`` in a MuJoCo-supported format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in MUJOCO_NATIVE_FORMATS:
        dest = out_dir / path.name
        if dest.resolve() != path.resolve():
            dest.write_bytes(path.read_bytes())
        return dest
    dest = out_dir / (path.stem + ".obj")
    logger.info("Converting %s to OBJ for MuJoCo", path.name)
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(
            [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    loaded.export(dest)
    return dest
