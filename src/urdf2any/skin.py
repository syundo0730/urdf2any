"""Skinned Collada (DAE) -> simulator-agnostic skin IR.

A skinned mesh binds vertices to skeleton bones; a robot converter needs the
same mesh expressed against *link* frames instead, so that a deformable visual
shell can ride on a body made of primitives. This module reads a DAE whose rest
pose matches the robot's zero configuration, bakes every vertex into world space
at that pose, and re-expresses the bone weights on link names. Backends turn the
result into MuJoCo ``<skin>`` assets (:mod:`urdf2any.backends.mjcf_skin`); the IR
is deliberately format-neutral so UsdSkel can be added later.

Conventions and known divergences:

- Bind poses come from the IR's zero-configuration FK
  (:meth:`~urdf2any.ir.Robot.link_world_transforms`), never from a compiled
  simulator model: the DAE must agree with the URDF, and this module verifies it
  rather than trusting it.
- The DAE's ``<unit>`` and ``<up_axis>`` asset metadata are applied on load, so
  baked vertices and bone rest positions are always SI meters, Z-up — the IR
  convention — regardless of how the document was authored.
- Texture coordinates are stored V-flipped (``v' = 1 - v``). Collada's UV origin
  is bottom-left; MuJoCo skins (and UsdSkel/glTF) use top-left.
- Vertices are de-duplicated on (position index, *rounded UV value*), not on UV
  index: Blender's exporter emits a distinct UV index per face corner, so keying
  on the index would split every shared vertex and flatten the shading.
- Mesh names strip Blender's ``Armature_<name>-skin`` controller-id decoration
  to recover the authored name; two controllers decoding to the same name are
  rejected rather than silently merged.
- Texture references resolve more leniently than the URDF parser's strict
  policy: after the document-relative path, the bare file name next to the DAE
  is tried, because exporters bake in absolute authoring-machine paths.

The bone -> link mapping is caller-supplied policy; nothing here guesses names.
Requires the optional ``pycollada`` dependency (``pip install "urdf2any[skin]"``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .ir import Robot
from .transforms import Transform

logger = logging.getLogger(__name__)

# Default tolerance [m] when checking a DAE bone's rest position against the
# forward kinematics of the link it is mapped to.
BIND_POSE_TOLERANCE = 1e-3

# UV values are compared at this many decimals when de-duplicating vertices.
_UV_DECIMALS = 6

# Color of a skinned mesh whose material has no texture and no diffuse color.
DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0])


@dataclass
class SkinnedMesh:
    """One skinned primitive: geometry baked into the bind pose, weighted by link."""

    name: str
    vertices: np.ndarray  # (N, 3) world coordinates at the bind pose
    faces: np.ndarray  # (M, 3) vertex indices
    uv: np.ndarray | None  # (N, 2), V-flipped (origin top-left)
    weights: dict[str, dict[int, float]]  # link name -> {vertex index: weight}
    texture: Path | None = None  # resolved image file, or None when flat-colored
    color: np.ndarray | None = None  # (4,) rgba, or None when textured

    def bone_links(self) -> list[str]:
        """Link names this mesh is weighted to, in a stable order."""
        return sorted(self.weights)


@dataclass
class SkinnedModel:
    """All skinned primitives of one DAE, plus the bind pose they were baked at."""

    meshes: list[SkinnedMesh]
    bind_poses: dict[str, Transform]  # link name -> world


def load_skinned_meshes(
    path: str | Path,
    robot: Robot,
    bone_links: dict[str, str],
    *,
    bind_pose_tolerance: float = BIND_POSE_TOLERANCE,
) -> SkinnedModel:
    """Load a skinned mesh file and bake it against the robot's zero-pose kinematics.

    Args:
        path: Skinned mesh file containing one or more skin controllers.
            Currently Collada (``.dae``) only; loaders for other skinned
            formats (e.g. glTF) can dispatch on the suffix here without
            changing this API.
        robot: Parsed IR. Its zero-configuration FK defines the bind pose, so it
            must be the same (possibly preprocessed) robot the backend emits.
        bone_links: DAE bone (joint sid) -> link name. Declarative and complete:
            every bone carrying a non-zero weight must be mapped. Bones absent
            from the DAE are ignored, so one mapping can serve several models.
        bind_pose_tolerance: Maximum allowed distance [m] between a bone's rest
            position in the DAE and the FK position of its link.

    Raises:
        ValueError: on a bind-pose mismatch, an unmapped weighted bone, a
            mapping that points at an unknown link, or unsupported DAE content.
        ImportError: if the optional ``pycollada`` dependency is missing.
    """
    path = Path(path)
    dae = _load_collada(path)
    bind_poses = robot.link_world_transforms()

    joint_world = _joint_world_transforms(dae)
    _verify_bind_pose(joint_world, bone_links, bind_poses, bind_pose_tolerance)

    model = SkinnedModel(
        meshes=_collect_meshes(dae, path, joint_world, bone_links),
        bind_poses=bind_poses,
    )
    logger.info(
        "Loaded %d skinned mesh(es) (%d vertices) from %s",
        len(model.meshes),
        sum(len(m.vertices) for m in model.meshes),
        path.name,
    )
    return model


def _load_collada(dae_path: Path):
    try:
        from collada import Collada
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "skin conversion requires the 'pycollada' package "
            "(pip install 'urdf2any[skin]')"
        ) from exc
    dae = Collada(str(dae_path))
    if dae.scene is None:
        raise ValueError(f"{dae_path} has no visual scene to read bone transforms from")
    return dae


# Rotations taking a document's up axis onto the IR's Z-up convention.
_UP_AXIS_ROTATIONS = {
    "Z_UP": np.eye(3),
    "Y_UP": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "X_UP": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
}


def _asset_transform(dae) -> np.ndarray:
    """4x4 mapping document space to SI meters, Z-up (the IR convention)."""
    info = dae.assetInfo
    scale = float(info.unitmeter) if info is not None and info.unitmeter else 1.0
    up_axis = info.upaxis if info is not None and info.upaxis else "Z_UP"
    rotation = _UP_AXIS_ROTATIONS.get(up_axis)
    if rotation is None:
        raise ValueError(f"unsupported Collada up_axis {up_axis!r}")
    transform = np.eye(4)
    transform[:3, :3] = rotation * scale
    return transform


def _joint_world_transforms(dae) -> dict[str, np.ndarray]:
    """Walk the scene graph: bone sid -> 4x4 world matrix at the DAE's rest pose.

    Seeded with the asset's unit/up-axis conversion, so the returned transforms
    (and everything skinned through them) are already SI, Z-up.
    """
    from collada.scene import Node

    world: dict[str, np.ndarray] = {}

    def walk(node, parent: np.ndarray) -> None:
        if not isinstance(node, Node):
            return
        transform = parent @ node.matrix
        sid = node.xmlnode.get("sid")
        if sid is not None:
            world[sid] = transform
        for child in getattr(node, "children", []):
            walk(child, transform)

    root = _asset_transform(dae)
    for node in dae.scene.nodes:
        walk(node, root)
    return world


def _verify_bind_pose(
    joint_world: dict[str, np.ndarray],
    bone_links: dict[str, str],
    bind_poses: dict[str, Transform],
    tolerance: float,
) -> None:
    """Check that the DAE rest pose agrees with the robot's zero-pose FK.

    Skinning bakes vertices at the DAE's rest pose but plays them back through
    link frames, so a disagreement silently deforms the model at runtime instead
    of failing. Positions only: a bone's roll about its own axis is not
    observable in the skeleton and need not match the link frame.
    """
    for bone, link in bone_links.items():
        if bone not in joint_world:
            continue  # mapping may cover several models; extra entries are fine
        if link not in bind_poses:
            raise ValueError(
                f"bone {bone!r} is mapped to link {link!r}, which is not in the robot "
                f"(was it merged away or renamed?)"
            )
        dae_pos = joint_world[bone][:3, 3]
        error = float(np.abs(dae_pos - bind_poses[link].translation).max())
        if error > tolerance:
            raise ValueError(
                f"bind pose mismatch for bone {bone!r} vs link {link!r}: {error:.6g} m "
                f"> {tolerance:.6g} m. The DAE rest pose and the URDF zero pose must agree."
            )


def _vertex_influences(skin) -> list[list[tuple[int, float]]]:
    """Per vertex: (joint index, raw weight) pairs, in document order.

    The one place that decodes pycollada's ragged weight arrays: joint index -1
    (the bind-shape "joint") and zero weights are dropped here, and a vertex
    left with no influence is an error. Weights stay raw; callers normalize.
    """
    influences: list[list[tuple[int, float]]] = []
    for vi in range(len(skin.joint_index)):
        pairs: list[tuple[int, float]] = []
        for j, w in zip(skin.joint_index[vi], skin.weight_index[vi], strict=True):
            weight = float(skin.weights[w][0])
            if j < 0 or weight == 0.0:
                continue
            pairs.append((int(j), weight))
        if not pairs:
            raise ValueError(f"vertex {vi} has no skinning weights")
        influences.append(pairs)
    return influences


def _bind_positions(
    skin, joint_world: dict[str, np.ndarray], influences: list[list[tuple[int, float]]]
) -> np.ndarray:
    """Skin the mesh into world space at the bind pose.

    ``v_bind = sum_j w_j * W_j * IBM_j * BSM * v`` with the weights normalized,
    where ``W_j`` is the bone's world transform from the scene graph, ``IBM_j``
    its inverse bind matrix and ``BSM`` the skin's bind shape matrix.
    """
    positions = np.asarray(skin.geometry.primitives[0].vertex, dtype=np.float64)
    composed = []
    for name in skin.weight_joints:
        if name not in joint_world:
            raise ValueError(f"joint node {name!r} is not in the DAE scene graph")
        composed.append(joint_world[name] @ skin.joint_matrices[name] @ skin.bind_shape_matrix)

    result = np.zeros_like(positions)
    for vi, pairs in enumerate(influences):
        homogeneous = np.append(positions[vi], 1.0)
        accum = np.zeros(3)
        total = 0.0
        for j, weight in pairs:
            accum += weight * (composed[j] @ homogeneous)[:3]
            total += weight
        result[vi] = accum / total
    return result


def _vertex_link_weights(
    skin, bone_links: dict[str, str], influences: list[list[tuple[int, float]]]
) -> list[dict[str, float]]:
    """Per vertex: link name -> normalized weight (weights on one link merged)."""
    joints = list(skin.weight_joints)
    per_vertex: list[dict[str, float]] = []
    for pairs in influences:
        weights: dict[str, float] = {}
        total = 0.0
        for j, weight in pairs:
            bone = joints[j]
            link = bone_links.get(bone)
            if link is None:
                raise ValueError(
                    f"bone {bone!r} carries skinning weights but is not in the bone -> link "
                    f"mapping; add it (mapped bones: {sorted(bone_links)})"
                )
            weights[link] = weights.get(link, 0.0) + weight
            total += weight
        per_vertex.append({link: w / total for link, w in weights.items()})
    return per_vertex


def _split_primitive(
    primitive, bind_positions: np.ndarray, vertex_weights: list[dict[str, float]]
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, dict[int, float]]]:
    """Emit per-corner vertices, merging corners that share a position and a UV value."""
    vertex_index = primitive.vertex_index.reshape(-1, 3)
    has_uv = bool(primitive.texcoord_indexset)
    uv_index = primitive.texcoord_indexset[0].reshape(-1, 3) if has_uv else None
    uv_source = np.asarray(primitive.texcoordset[0], dtype=np.float64) if has_uv else None

    key_to_index: dict[tuple[int, float, float], int] = {}
    vertices: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    faces = np.empty_like(vertex_index)
    weights: dict[str, dict[int, float]] = {}
    for ti in range(len(vertex_index)):
        for corner in range(3):
            vi = int(vertex_index[ti, corner])
            if uv_index is not None:
                ui = int(uv_index[ti, corner])
                uv = (float(uv_source[ui][0]), 1.0 - float(uv_source[ui][1]))
                key = (vi, round(uv[0], _UV_DECIMALS), round(uv[1], _UV_DECIMALS))
            else:
                uv = None
                key = (vi, 0.0, 0.0)
            new_index = key_to_index.get(key)
            if new_index is None:
                new_index = len(vertices)
                key_to_index[key] = new_index
                vertices.append(bind_positions[vi])
                if uv is not None:
                    uvs.append(np.asarray(uv))
                for link, weight in vertex_weights[vi].items():
                    weights.setdefault(link, {})[new_index] = weight
            faces[ti, corner] = new_index
    return np.asarray(vertices), (np.asarray(uvs) if uvs else None), faces, weights


def _material_bindings(dae) -> dict[str, dict[str, object]]:
    """Controller id -> {material symbol: material}, from instance_controller bindings."""
    from collada.scene import ControllerNode

    bindings: dict[str, dict[str, object]] = {}

    def walk(node) -> None:
        if isinstance(node, ControllerNode):
            bindings[node.controller.id] = {m.symbol: m.target for m in node.materials}
        for child in getattr(node, "children", []):
            walk(child)

    for node in dae.scene.nodes:
        walk(node)
    return bindings


def _resolve_texture(image_path: str, dae_dir: Path) -> Path:
    """Resolve an ``<init_from>`` image reference against the DAE's directory.

    The raw value may be percent-encoded and may carry subdirectories. Exporters
    also bake in absolute paths from the authoring machine, so when the full
    path does not resolve, fall back to the bare file name next to the DAE.
    """
    raw = unquote(image_path).removeprefix("file://")
    path = Path(raw)
    # An absolute path is deliberately never tried as-is: it points into the
    # authoring machine's filesystem, and only its basename is meaningful here.
    candidates = [] if path.is_absolute() else [dae_dir / path]
    candidates.append(dae_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"texture {raw!r} not found relative to {dae_dir}")


def _appearance(
    material_symbol: str, bindings: dict[str, object], dae_dir: Path
) -> tuple[Path | None, np.ndarray | None]:
    """(texture file, rgba) for a primitive's bound material; exactly one is set."""
    material = bindings.get(material_symbol)
    diffuse = material.effect.diffuse if material is not None else None
    if diffuse is not None and hasattr(diffuse, "sampler"):
        return _resolve_texture(diffuse.sampler.surface.image.path, dae_dir), None
    if isinstance(diffuse, tuple):
        return None, np.asarray(diffuse, dtype=float)
    return None, DEFAULT_RGBA.copy()


def _mesh_name(controller_id: str, primitive_index: int) -> str:
    # Blender's Collada exporter decorates skin controller ids as
    # "Armature_<mesh>-skin"; strip that to get back the authored mesh name.
    base = controller_id.removeprefix("Armature_").removesuffix("-skin")
    return base if primitive_index == 0 else f"{base}_{primitive_index}"


def _collect_meshes(
    dae,
    dae_path: Path,
    joint_world: dict[str, np.ndarray],
    bone_links: dict[str, str],
) -> list[SkinnedMesh]:
    from collada.controller import Skin
    from collada.triangleset import TriangleSet

    bindings = _material_bindings(dae)
    meshes: list[SkinnedMesh] = []
    seen_names: set[str] = set()
    for controller in dae.controllers:
        if not isinstance(controller, Skin):
            continue
        influences = _vertex_influences(controller)
        positions = _bind_positions(controller, joint_world, influences)
        # Weighted bones are always in the scene graph (_bind_positions checks)
        # and therefore already validated against the robot by _verify_bind_pose.
        vertex_weights = _vertex_link_weights(controller, bone_links, influences)
        controller_bindings = bindings.get(controller.id, {})
        for index, primitive in enumerate(controller.geometry.primitives):
            if not isinstance(primitive, TriangleSet):
                raise ValueError(
                    f"unsupported primitive {type(primitive).__name__} in {controller.id!r} "
                    "(skins must be triangulated)"
                )
            if len(primitive.vertex) != len(positions):
                # Skin weights are per-vertex of the geometry's first position
                # source; a primitive on a different source cannot reuse them.
                raise ValueError(
                    f"primitive {index} of {controller.id!r} uses a different position "
                    "source than the skinned one (unsupported)"
                )
            name = _mesh_name(controller.id, index)
            if name in seen_names:
                # Stripping exporter decoration can collapse distinct controller
                # ids; failing here beats silently overwriting a .skn file.
                raise ValueError(
                    f"duplicate skinned mesh name {name!r} (from controller {controller.id!r})"
                )
            seen_names.add(name)
            vertices, uv, faces, weights = _split_primitive(primitive, positions, vertex_weights)
            texture, color = _appearance(primitive.material, controller_bindings, dae_path.parent)
            meshes.append(
                SkinnedMesh(
                    name=name,
                    vertices=vertices,
                    faces=faces,
                    uv=uv,
                    weights=weights,
                    texture=texture,
                    color=color,
                )
            )
    if not meshes:
        raise ValueError(f"{dae_path} contains no skin controllers")
    return meshes


__all__ = [
    "BIND_POSE_TOLERANCE",
    "SkinnedMesh",
    "SkinnedModel",
    "load_skinned_meshes",
]
