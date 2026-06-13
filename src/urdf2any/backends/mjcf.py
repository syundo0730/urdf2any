"""URDF IR -> MJCF (MuJoCo XML) conversion by direct XML authoring.

MuJoCo's native URDF loader is deliberately not used: its output is a flattened
"canonical" model with no actuators and no <default> classes. Authoring the XML
directly lets us emit a readable, actuator-equipped model.

Unit conventions: the model uses ``<compiler angle="radian"/>``, so all joint
ranges and gains are written in SI/radian, matching the IR directly.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np

from ..config import ConvertConfig
from ..ir import (
    Box,
    Capsule,
    Cylinder,
    Geometry,
    Joint,
    Link,
    MeshRef,
    Robot,
    Sphere,
)
from ..mesh import ensure_mujoco_compatible

logger = logging.getLogger(__name__)

# MuJoCo reserves "world" for the implicit worldbody; a URDF link of that name
# would collide ("repeated name 'world' in body"), so it is renamed on output.
MJCF_RESERVED_BODY_NAMES = {"world"}


def _safe_body_names(robot: Robot) -> dict[str, str]:
    """Map link names to MuJoCo-safe body names (avoiding reserved names)."""
    names: dict[str, str] = {}
    for link_name in robot.links:
        if link_name not in MJCF_RESERVED_BODY_NAMES:
            names[link_name] = link_name
            continue
        candidate = f"{link_name}_link"
        i = 1
        while candidate in robot.links or candidate in names.values():
            i += 1
            candidate = f"{link_name}_link_{i}"
        names[link_name] = candidate
        logger.warning(
            "MJCF: renamed link %r to %r (collides with a reserved MuJoCo body name)",
            link_name,
            candidate,
        )
    return names


def convert_mjcf(robot: Robot, output_path: str | Path, config: ConvertConfig) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = output_path.parent / config.mjcf_assets_dir

    writer = _MjcfWriter(robot, config, assets_dir)
    root = writer.build()

    xml = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    # Drop blank lines minidom inserts.
    xml = "\n".join(line for line in xml.splitlines() if line.strip())
    output_path.write_text(xml + "\n", encoding="utf-8")
    logger.info("Wrote %s", output_path)
    return output_path


def _fmt(values) -> str:
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    return " ".join(f"{v:.9g}" for v in arr)


def _quat_str(q: np.ndarray) -> str:
    return _fmt(q)  # MuJoCo quaternions are (w, x, y, z), same as the IR


class _MjcfWriter:
    def __init__(self, robot: Robot, config: ConvertConfig, assets_dir: Path):
        self.robot = robot
        self.config = config
        self.assets_dir = assets_dir
        # Keyed by (path, scale): MuJoCo mesh scale is per-asset, so the same file
        # referenced at different scales becomes distinct mesh assets.
        self.meshes: dict[tuple[str, tuple[float, float, float]], tuple[str, MeshRef]] = {}
        self.material_names: set[str] = set()
        self.children = robot.child_joints()
        self.actuated: list[Joint] = []
        self.body_names = _safe_body_names(robot)

    def build(self) -> ET.Element:
        root = ET.Element("mujoco", model=self.robot.name)
        ET.SubElement(
            root, "compiler", angle="radian", meshdir=self.config.mjcf_assets_dir, autolimits="true"
        )

        asset = ET.SubElement(root, "asset")
        worldbody = ET.SubElement(root, "worldbody")

        root_link = self.robot.root_link_name()
        self._write_body(worldbody, root_link, is_root=True)

        self._write_assets(asset)

        if self.config.mjcf_add_actuators and self.actuated:
            self._write_actuators(root)
        return root

    # ----- bodies -----------------------------------------------------------

    def _write_body(self, parent_elem: ET.Element, link_name: str, is_root: bool) -> None:
        link = self.robot.links[link_name]
        parent_joint = self.robot.parent_joint(link_name)

        body = ET.SubElement(parent_elem, "body", name=self.body_names[link_name])
        if parent_joint is not None:
            origin = parent_joint.origin
            if np.any(origin.translation):
                body.set("pos", _fmt(origin.translation))
            if not _is_identity_quat(origin.rotation):
                body.set("quat", _quat_str(origin.rotation))

        if is_root and not self.config.fix_base:
            ET.SubElement(body, "freejoint")

        if parent_joint is not None and parent_joint.type != "fixed":
            self._write_joint(body, parent_joint)

        self._write_inertial(body, link)

        for i, geometry in enumerate(link.visuals):
            self._write_geom(body, geometry, i, collision=False)
        for i, geometry in enumerate(link.collisions):
            self._write_geom(body, geometry, i, collision=True)

        for joint in self.children[link_name]:
            self._write_body(body, joint.child, is_root=False)

    def _write_inertial(self, body: ET.Element, link: Link) -> None:
        inertial = link.inertial
        if inertial is None or inertial.mass is None:
            # Let MuJoCo infer inertia from geometry when mass is unknown.
            return
        attrs = {"mass": _fmt(inertial.mass)}
        com = inertial.origin.translation if inertial.origin is not None else np.zeros(3)
        attrs["pos"] = _fmt(com)
        if inertial.inertia is not None:
            m = inertial.inertia
            # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
            attrs["fullinertia"] = _fmt(
                [m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]]
            )
        else:
            # Mass known but no inertia: give a tiny diagonal so MuJoCo accepts it;
            # exact value is refined by geometry if present.
            attrs["diaginertia"] = _fmt([1e-6, 1e-6, 1e-6])
        ET.SubElement(body, "inertial", **attrs)

    def _write_joint(self, body: ET.Element, joint: Joint) -> None:
        attrs = {"name": joint.name, "axis": _fmt(joint.axis)}
        if joint.type in ("revolute", "continuous"):
            attrs["type"] = "hinge"
        elif joint.type == "prismatic":
            attrs["type"] = "slide"
        if joint.type != "continuous" and joint.limit is not None:
            attrs["range"] = _fmt([joint.limit.lower, joint.limit.upper])
        if joint.dynamics is not None and joint.dynamics.damping > 0:
            attrs["damping"] = _fmt(joint.dynamics.damping)
        if joint.dynamics is not None and joint.dynamics.friction > 0:
            attrs["frictionloss"] = _fmt(joint.dynamics.friction)
        ET.SubElement(body, "joint", **attrs)
        if joint.drive is not None and joint.drive.target_type != "none":
            self.actuated.append(joint)

    # ----- geometry ---------------------------------------------------------

    def _write_geom(self, body: ET.Element, geometry: Geometry, index: int, collision: bool) -> None:
        shape = geometry.shape
        attrs: dict[str, str] = {}
        name = geometry.name or f"{'col' if collision else 'vis'}_{body.get('name')}_{index}"
        attrs["name"] = name

        origin = geometry.origin
        if np.any(origin.translation):
            attrs["pos"] = _fmt(origin.translation)
        if not _is_identity_quat(origin.rotation):
            attrs["quat"] = _quat_str(origin.rotation)

        if isinstance(shape, Box):
            attrs["type"] = "box"
            attrs["size"] = _fmt(shape.size / 2.0)  # MuJoCo box size is half-extents
        elif isinstance(shape, Sphere):
            attrs["type"] = "sphere"
            attrs["size"] = _fmt(shape.radius)
        elif isinstance(shape, Cylinder):
            attrs["type"] = "cylinder"
            attrs["size"] = _fmt([shape.radius, shape.length / 2.0])
        elif isinstance(shape, Capsule):
            attrs["type"] = "capsule"
            attrs["size"] = _fmt([shape.radius, shape.length / 2.0])
        elif isinstance(shape, MeshRef):
            attrs["type"] = "mesh"
            attrs["mesh"] = self._register_mesh(shape)
        else:  # pragma: no cover
            raise TypeError(f"Unsupported shape: {shape}")

        # Convention from mujoco_menagerie: visual geoms render-only (group 2),
        # collision geoms collide but are hidden (group 3).
        if collision:
            attrs["group"] = "3"
        else:
            attrs["group"] = "2"
            attrs["contype"] = "0"
            attrs["conaffinity"] = "0"
            if geometry.material:
                attrs["material"] = self._register_material(geometry.material)

        ET.SubElement(body, "geom", **attrs)

    def _register_mesh(self, ref: MeshRef) -> str:
        key = (str(ref.path), tuple(float(s) for s in ref.scale))
        if key in self.meshes:
            return self.meshes[key][0]
        name = ref.path.stem
        # Disambiguate collisions (same stem from a different dir or scale).
        existing = {n for n, _ in self.meshes.values()}
        suffix = 1
        while name in existing:
            suffix += 1
            name = f"{ref.path.stem}_{suffix}"
        self.meshes[key] = (name, ref)
        return name

    def _register_material(self, name: str) -> str:
        self.material_names.add(name)
        return name

    # ----- assets / actuators -----------------------------------------------

    def _write_assets(self, asset: ET.Element) -> None:
        for name, ref in self.meshes.values():
            dest = ensure_mujoco_compatible(ref.path, self.assets_dir)
            attrs = {"name": name, "file": dest.name}
            if not np.allclose(ref.scale, 1.0):
                attrs["scale"] = _fmt(ref.scale)
            ET.SubElement(asset, "mesh", **attrs)

        for name in sorted(self.material_names):
            material = self.robot.materials.get(name)
            attrs = {"name": name}
            if material is not None and material.color is not None:
                attrs["rgba"] = _fmt(material.color)
            ET.SubElement(asset, "material", **attrs)

    def _write_actuators(self, root: ET.Element) -> None:
        actuator = ET.SubElement(root, "actuator")
        for joint in self.actuated:
            drive = joint.drive
            assert drive is not None
            if drive.target_type == "position":
                attrs = {
                    "name": joint.name,
                    "joint": joint.name,
                    "kp": _fmt(drive.stiffness),
                }
                if drive.damping > 0:
                    attrs["kv"] = _fmt(drive.damping)
                if joint.limit is not None and joint.limit.effort > 0:
                    attrs["forcerange"] = _fmt([-joint.limit.effort, joint.limit.effort])
                ET.SubElement(actuator, "position", **attrs)
            elif drive.target_type == "velocity":
                attrs = {"name": joint.name, "joint": joint.name, "kv": _fmt(drive.stiffness)}
                if joint.limit is not None and joint.limit.effort > 0:
                    attrs["forcerange"] = _fmt([-joint.limit.effort, joint.limit.effort])
                ET.SubElement(actuator, "velocity", **attrs)


def _is_identity_quat(q: np.ndarray) -> bool:
    return bool(np.allclose(q, [1.0, 0.0, 0.0, 0.0]))
