"""Simulator-agnostic intermediate representation of a robot.

All quantities are SI: meters, kilograms, radians, N·m. Unit conversions
(e.g. USD's degree-based joint attributes) are the responsibility of backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .transforms import Transform

JointType = Literal["revolute", "continuous", "prismatic", "fixed"]
DriveTargetType = Literal["position", "velocity", "none"]
DriveType = Literal["force", "acceleration"]


@dataclass
class Box:
    size: np.ndarray  # (3,) full extents


@dataclass
class Cylinder:
    radius: float
    length: float  # along local Z


@dataclass
class Sphere:
    radius: float


@dataclass
class Capsule:
    radius: float
    length: float  # cylindrical section, along local Z


@dataclass
class MeshRef:
    path: Path  # resolved absolute path
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))


Shape = Box | Cylinder | Sphere | Capsule | MeshRef


@dataclass
class Geometry:
    shape: Shape
    origin: Transform = field(default_factory=Transform)
    material: str | None = None
    name: str | None = None


@dataclass
class Inertial:
    mass: float | None = None
    origin: Transform | None = None  # center of mass frame
    inertia: np.ndarray | None = None  # 3x3 tensor about the COM, in the origin frame


@dataclass
class JointLimit:
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0


@dataclass
class JointDynamics:
    damping: float = 0.0
    friction: float = 0.0


@dataclass
class JointMimic:
    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass
class JointDrive:
    """Actuation gains. Not part of URDF; injected from ConvertConfig."""

    target_type: DriveTargetType = "position"
    drive_type: DriveType = "force"
    stiffness: float = 0.0  # N·m/rad or N/m
    damping: float = 0.0  # N·m·s/rad or N·s/m


@dataclass
class Joint:
    name: str
    type: JointType
    parent: str
    child: str
    origin: Transform = field(default_factory=Transform)
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    limit: JointLimit | None = None
    dynamics: JointDynamics | None = None
    mimic: JointMimic | None = None
    drive: JointDrive | None = None


@dataclass
class Material:
    name: str
    color: np.ndarray | None = None  # (4,) rgba
    texture: Path | None = None


@dataclass
class Link:
    name: str
    inertial: Inertial | None = None
    visuals: list[Geometry] = field(default_factory=list)
    collisions: list[Geometry] = field(default_factory=list)


@dataclass
class Robot:
    name: str
    links: dict[str, Link] = field(default_factory=dict)
    joints: dict[str, Joint] = field(default_factory=dict)
    materials: dict[str, Material] = field(default_factory=dict)

    def root_link_name(self) -> str:
        children = {j.child for j in self.joints.values()}
        roots = [name for name in self.links if name not in children]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one root link, found: {roots}")
        return roots[0]

    def child_joints(self) -> dict[str, list[Joint]]:
        """Map of link name -> joints whose parent is that link, in declaration order."""
        out: dict[str, list[Joint]] = {name: [] for name in self.links}
        for joint in self.joints.values():
            if joint.parent not in self.links:
                raise ValueError(f"Joint {joint.name} references unknown parent link {joint.parent}")
            if joint.child not in self.links:
                raise ValueError(f"Joint {joint.name} references unknown child link {joint.child}")
            out[joint.parent].append(joint)
        return out

    def parent_joint(self, link_name: str) -> Joint | None:
        for joint in self.joints.values():
            if joint.child == link_name:
                return joint
        return None

    def link_world_transforms(self) -> dict[str, Transform]:
        """Zero-configuration forward kinematics: link name -> world transform."""
        root = self.root_link_name()
        children = self.child_joints()
        world: dict[str, Transform] = {root: Transform.identity()}
        stack = [root]
        while stack:
            parent = stack.pop()
            for joint in children[parent]:
                if joint.child in world:
                    raise ValueError(f"Kinematic loop detected at link {joint.child}")
                world[joint.child] = world[parent] @ joint.origin
                stack.append(joint.child)
        missing = set(self.links) - set(world)
        if missing:
            raise ValueError(f"Links not connected to the kinematic tree: {sorted(missing)}")
        return world


def frame_only_links(robot: Robot) -> list[Link]:
    """Links with no visual and no collision geometry (URDF frame markers)."""
    return [link for link in robot.links.values() if not link.visuals and not link.collisions]
