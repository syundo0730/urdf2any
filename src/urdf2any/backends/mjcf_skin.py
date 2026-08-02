"""Skin IR -> MuJoCo ``.skn`` binary assets.

MuJoCo keeps skins out of the model XML: ``<skin file="x.skn"/>`` points at a
standalone little-endian binary laid out as

    int32   nvertex, ntexcoord, nface, nbone
    float32 vertex[3 * nvertex]          world coordinates at the bind pose
    float32 texcoord[2 * ntexcoord]
    int32   face[3 * nface]
    per bone:
        char    name[40]                 body (link) name, NUL-padded
        float32 bindpos[3]
        float32 bindquat[4]              (w, x, y, z)
        int32   vertexnum
        int32   vertexid[vertexnum]
        float32 vertexweight[vertexnum]

This module writes those files, copies the textures they need, and returns what
the caller needs to author the ``<asset>`` block. Where the assets go in a model,
how they are named and how the ``<skin>`` elements get into the model file are
caller policy and stay out of here.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..skin import SkinnedMesh, SkinnedModel
from ..transforms import Transform

logger = logging.getLogger(__name__)

# Fixed-width bone name field in the .skn format.
BONE_NAME_BYTES = 40


@dataclass
class SkinAsset:
    """One written skin, described well enough to author its ``<asset>`` XML."""

    name: str
    file: str  # .skn file name, inside the output directory
    texture_file: str | None  # copied image file name, or None when flat-colored
    rgba: tuple[float, float, float, float] | None  # set iff texture_file is None
    bones: tuple[str, ...]  # link names the skin is bound to
    vertex_count: int


def write_mjcf_skins(
    model: SkinnedModel,
    out_dir: str | Path,
    *,
    name_prefix: str = "",
    body_names: dict[str, str] | None = None,
) -> list[SkinAsset]:
    """Write one ``.skn`` per skinned mesh into ``out_dir``, plus its textures.

    Args:
        model: Result of :func:`urdf2any.skin.load_skinned_meshes`.
        out_dir: Directory for the ``.skn`` files and texture copies; created if
            missing. Typically the MJCF model's asset directory.
        name_prefix: Prepended (with an underscore) to every skin name, to keep
            several robots' skins apart in one asset namespace.
        body_names: Link name -> emitted MJCF body name. Required whenever the
            model's body names differ from the IR's link names — ``fix_base``
            folds the root link into the worldbody ("world"), and reserved link
            names are renamed. Get it from
            :func:`urdf2any.backends.mjcf.mjcf_body_names`. Links absent from
            the map keep their own name.

    Returns:
        One :class:`SkinAsset` per mesh, in model order. ``bones`` holds the
        translated body names actually written to the files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body_names = body_names or {}

    assets: list[SkinAsset] = []
    copied_textures: dict[str, Path] = {}
    for mesh in model.meshes:
        name = f"{name_prefix}_{mesh.name}" if name_prefix else mesh.name
        path = out_dir / f"{name}.skn"
        _write_skn(path, mesh, model.bind_poses, body_names)
        texture_file = _copy_texture(mesh.texture, out_dir, copied_textures) if mesh.texture else None
        assets.append(
            SkinAsset(
                name=name,
                file=path.name,
                texture_file=texture_file,
                rgba=None if texture_file else _rgba(mesh),
                bones=tuple(body_names.get(link, link) for link in mesh.bone_links()),
                vertex_count=len(mesh.vertices),
            )
        )
    logger.info(
        "Wrote %d skin(s) (%d vertices) to %s",
        len(assets),
        sum(a.vertex_count for a in assets),
        out_dir,
    )
    return assets


def _rgba(mesh: SkinnedMesh) -> tuple[float, float, float, float]:
    color = mesh.color if mesh.color is not None else np.array([0.5, 0.5, 0.5, 1.0])
    values = tuple(float(v) for v in color)
    return values if len(values) == 4 else (*values[:3], 1.0)


def _copy_texture(texture: Path, out_dir: Path, copied: dict[str, Path]) -> str:
    source = texture.resolve()
    prior = copied.get(texture.name)
    if prior is not None and prior != source:
        # Two textures collapsing to one basename would silently shadow each
        # other in the flat output directory.
        raise ValueError(f"texture name collision: {prior} and {source} both copy to {texture.name}")
    copied[texture.name] = source
    dest = out_dir / texture.name
    if dest.resolve() != source:
        dest.write_bytes(texture.read_bytes())
    return dest.name


def _write_skn(
    path: Path, mesh: SkinnedMesh, bind_poses: dict[str, Transform], body_names: dict[str, str]
) -> None:
    links = mesh.bone_links()
    ntexcoord = len(mesh.uv) if mesh.uv is not None else 0
    with path.open("wb") as f:
        f.write(struct.pack("<4i", len(mesh.vertices), ntexcoord, len(mesh.faces), len(links)))
        f.write(mesh.vertices.astype("<f4").tobytes())
        if mesh.uv is not None:
            f.write(mesh.uv.astype("<f4").tobytes())
        f.write(mesh.faces.astype("<i4").tobytes())
        for link in links:
            pose = bind_poses[link]
            body = body_names.get(link, link)
            name = body.encode("utf-8")
            if len(name) >= BONE_NAME_BYTES:
                raise ValueError(
                    f"body name {body!r} does not fit the skn bone name field "
                    f"({BONE_NAME_BYTES - 1} bytes)"
                )
            f.write(name.ljust(BONE_NAME_BYTES, b"\x00"))
            f.write(np.asarray(pose.translation, dtype="<f4").tobytes())
            f.write(np.asarray(pose.rotation, dtype="<f4").tobytes())
            weights = mesh.weights[link]
            vertex_ids = sorted(weights)
            f.write(struct.pack("<i", len(vertex_ids)))
            f.write(np.asarray(vertex_ids, dtype="<i4").tobytes())
            f.write(np.asarray([weights[i] for i in vertex_ids], dtype="<f4").tobytes())
