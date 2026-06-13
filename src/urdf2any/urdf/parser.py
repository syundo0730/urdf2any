"""URDF file parser producing the urdf2any intermediate representation."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..ir import (
    Box,
    Capsule,
    Cylinder,
    Geometry,
    Inertial,
    Joint,
    JointDynamics,
    JointLimit,
    JointMimic,
    Link,
    Material,
    MeshRef,
    Robot,
    Sphere,
)
from ..transforms import Transform, quat_from_rpy

logger = logging.getLogger(__name__)

SUPPORTED_JOINT_TYPES = {"revolute", "continuous", "prismatic", "fixed"}


def parse_urdf(path: str | Path, mesh_root: str | Path | None = None) -> Robot:
    """Parse a URDF file into a :class:`Robot`.

    Args:
        path: URDF file path.
        mesh_root: Base directory for resolving ``package://`` mesh URIs.
            Defaults to the URDF file's directory.
    """
    path = Path(path).resolve()
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected URDF root tag 'robot', got {root.tag!r}")

    urdf_dir = path.parent
    mesh_root_dir = Path(mesh_root).resolve() if mesh_root else urdf_dir

    robot = Robot(name=root.get("name", path.stem))

    for elem in root.findall("material"):
        material = _parse_material(elem, urdf_dir)
        if material is not None:
            robot.materials[material.name] = material

    for elem in root.findall("link"):
        link = _parse_link(elem, urdf_dir, mesh_root_dir, robot)
        robot.links[link.name] = link

    for elem in root.findall("joint"):
        joint = _parse_joint(elem)
        if joint is not None:
            robot.joints[joint.name] = joint

    # Validate tree connectivity early so backends can assume a well-formed model.
    robot.root_link_name()
    robot.link_world_transforms()
    return robot


def _floats(text: str | None, n: int, default: list[float] | None = None) -> np.ndarray:
    if text is None:
        if default is None:
            raise ValueError("Missing required numeric attribute")
        return np.array(default, dtype=float)
    values = [float(v) for v in text.split()]
    if len(values) != n:
        raise ValueError(f"Expected {n} values, got {text!r}")
    return np.array(values)


def _parse_origin(elem: ET.Element | None) -> Transform:
    if elem is None:
        return Transform.identity()
    xyz = _floats(elem.get("xyz"), 3, [0.0, 0.0, 0.0])
    rpy = _floats(elem.get("rpy"), 3, [0.0, 0.0, 0.0])
    return Transform(translation=xyz, rotation=quat_from_rpy(*rpy))


def _parse_material(elem: ET.Element, urdf_dir: Path) -> Material | None:
    name = elem.get("name", "")
    color_elem = elem.find("color")
    texture_elem = elem.find("texture")
    color = _floats(color_elem.get("rgba"), 4) if color_elem is not None else None
    texture = None
    if texture_elem is not None and texture_elem.get("filename"):
        texture = _resolve_path(texture_elem.get("filename", ""), urdf_dir, urdf_dir)
    if color is None and texture is None:
        return Material(name=name) if name else None
    return Material(name=name, color=color, texture=texture)


def _resolve_path(filename: str, urdf_dir: Path, mesh_root: Path) -> Path:
    """Resolve a URDF asset path (plain, relative, or package:// URI)."""
    if filename.startswith("package://"):
        # package://<pkg_name>/<rel_path>: search for <rel_path> under mesh_root,
        # then under mesh_root/<pkg_name>.
        without_scheme = filename[len("package://") :]
        pkg, _, rel = without_scheme.partition("/")
        for candidate in (mesh_root / rel, mesh_root / pkg / rel, urdf_dir / rel):
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f"Cannot resolve {filename!r} (searched under {mesh_root})")
    if filename.startswith("file://"):
        filename = filename[len("file://") :]
    p = Path(filename)
    if not p.is_absolute():
        p = urdf_dir / p
    if not p.exists():
        raise FileNotFoundError(f"Mesh file not found: {p}")
    return p.resolve()


def _parse_geometry(
    elem: ET.Element, urdf_dir: Path, mesh_root: Path, robot: Robot
) -> Geometry | None:
    geom_elem = elem.find("geometry")
    if geom_elem is None:
        return None
    shape_elem = geom_elem[0]
    match shape_elem.tag:
        case "box":
            shape = Box(size=_floats(shape_elem.get("size"), 3))
        case "cylinder":
            shape = Cylinder(
                radius=float(shape_elem.get("radius", 0)), length=float(shape_elem.get("length", 0))
            )
        case "sphere":
            shape = Sphere(radius=float(shape_elem.get("radius", 0)))
        case "capsule":
            shape = Capsule(
                radius=float(shape_elem.get("radius", 0)), length=float(shape_elem.get("length", 0))
            )
        case "mesh":
            shape = MeshRef(
                path=_resolve_path(shape_elem.get("filename", ""), urdf_dir, mesh_root),
                scale=_floats(shape_elem.get("scale"), 3, [1.0, 1.0, 1.0]),
            )
        case other:
            logger.warning("Skipping unsupported geometry type %r", other)
            return None

    material_name = None
    material_elem = elem.find("material")
    if material_elem is not None:
        material_name = material_elem.get("name") or None
        # Inline material definitions register into the robot-wide table.
        if material_name and material_name not in robot.materials:
            inline = _parse_material(material_elem, urdf_dir)
            if inline is not None and (inline.color is not None or inline.texture is not None):
                robot.materials[material_name] = inline

    return Geometry(
        shape=shape,
        origin=_parse_origin(elem.find("origin")),
        material=material_name,
        name=elem.get("name"),
    )


def _parse_inertial(elem: ET.Element | None) -> Inertial | None:
    if elem is None:
        return None
    inertial = Inertial()
    mass_elem = elem.find("mass")
    if mass_elem is not None:
        inertial.mass = float(mass_elem.get("value", 0))
    origin_elem = elem.find("origin")
    if origin_elem is not None:
        inertial.origin = _parse_origin(origin_elem)
    inertia_elem = elem.find("inertia")
    if inertia_elem is not None:
        ixx = float(inertia_elem.get("ixx", 0))
        ixy = float(inertia_elem.get("ixy", 0))
        ixz = float(inertia_elem.get("ixz", 0))
        iyy = float(inertia_elem.get("iyy", 0))
        iyz = float(inertia_elem.get("iyz", 0))
        izz = float(inertia_elem.get("izz", 0))
        inertial.inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    if inertial.mass is None and inertial.inertia is None:
        return None
    return inertial


def _parse_link(elem: ET.Element, urdf_dir: Path, mesh_root: Path, robot: Robot) -> Link:
    name = elem.get("name")
    if not name:
        raise ValueError("Link without a name")
    link = Link(name=name, inertial=_parse_inertial(elem.find("inertial")))
    for visual in elem.findall("visual"):
        geometry = _parse_geometry(visual, urdf_dir, mesh_root, robot)
        if geometry is not None:
            link.visuals.append(geometry)
    for collision in elem.findall("collision"):
        geometry = _parse_geometry(collision, urdf_dir, mesh_root, robot)
        if geometry is not None:
            link.collisions.append(geometry)
    return link


def _parse_joint(elem: ET.Element) -> Joint | None:
    name = elem.get("name")
    joint_type = elem.get("type")
    if not name or not joint_type:
        raise ValueError("Joint without name or type")
    if joint_type not in SUPPORTED_JOINT_TYPES:
        raise ValueError(
            f"Joint {name!r} has unsupported type {joint_type!r} "
            f"(supported: {sorted(SUPPORTED_JOINT_TYPES)})"
        )

    parent_elem = elem.find("parent")
    child_elem = elem.find("child")
    if parent_elem is None or child_elem is None:
        raise ValueError(f"Joint {name!r} is missing parent or child")

    joint = Joint(
        name=name,
        type=joint_type,  # type: ignore[arg-type]
        parent=parent_elem.get("link", ""),
        child=child_elem.get("link", ""),
        origin=_parse_origin(elem.find("origin")),
    )

    axis_elem = elem.find("axis")
    if axis_elem is not None:
        joint.axis = _floats(axis_elem.get("xyz"), 3, [1.0, 0.0, 0.0])

    limit_elem = elem.find("limit")
    if limit_elem is not None:
        joint.limit = JointLimit(
            lower=float(limit_elem.get("lower", 0)),
            upper=float(limit_elem.get("upper", 0)),
            effort=float(limit_elem.get("effort", 0)),
            velocity=float(limit_elem.get("velocity", 0)),
        )
    elif joint_type in ("revolute", "prismatic"):
        raise ValueError(f"Joint {name!r} of type {joint_type!r} requires a <limit> element")

    dynamics_elem = elem.find("dynamics")
    if dynamics_elem is not None:
        joint.dynamics = JointDynamics(
            damping=float(dynamics_elem.get("damping", 0)),
            friction=float(dynamics_elem.get("friction", 0)),
        )

    mimic_elem = elem.find("mimic")
    if mimic_elem is not None:
        joint.mimic = JointMimic(
            joint=mimic_elem.get("joint", ""),
            multiplier=float(mimic_elem.get("multiplier", 1)),
            offset=float(mimic_elem.get("offset", 0)),
        )

    return joint
