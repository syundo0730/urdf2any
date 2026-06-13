"""Conversion configuration and generic IR preprocessing.

Policy (which joints get which gains, which geometry to drop) lives in the
caller's configuration; this module only provides the mechanisms. Field
vocabulary intentionally mirrors IsaacLab's ``UrdfConverterCfg`` to ease
migration.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from .ir import Capsule, Cylinder, Inertial, JointDrive, Robot, frame_only_links

logger = logging.getLogger(__name__)

# A scalar applies to all joints; a dict maps joint-name regex -> value.
PerJoint = float | dict[str, float]
PerJointStr = str | dict[str, str]


@dataclass
class JointDriveCfg:
    target_type: PerJointStr = "position"  # position | velocity | none
    drive_type: PerJointStr = "force"  # force | acceleration
    stiffness: PerJoint = 100.0  # N·m/rad (revolute) or N/m (prismatic)
    damping: PerJoint = 1.0


@dataclass
class ConvertConfig:
    fix_base: bool = False
    self_collision: bool = False
    collider_type: str = "convex_hull"  # convex_hull | convex_decomposition
    replace_cylinders_with_capsules: bool = False
    joint_drive: JointDriveCfg | None = field(default_factory=JointDriveCfg)
    # Set a small inertial on frame-only links (no visual/collision), overwriting
    # any authored inertial. None disables.
    frame_link_mass: float | None = None
    frame_link_inertia: float | None = None
    # Omit authored mass / inertia from the USD for links that have collision
    # geometry, leaving the consumer (e.g. PhysX, MuJoCo) to compute them from the
    # collision shapes. Scoped to collision links because that is the only case
    # the consumer can recompute from. skip_collision_inertia drops the inertia
    # tensor + COM origin (keeping mass); skip_collision_mass drops the mass.
    skip_collision_inertia: bool = False
    skip_collision_mass: bool = False
    # Drop visual/collision geometry whose name matches any of these regexes.
    exclude_geometry_names: list[str] = field(default_factory=list)
    # MJCF backend
    mjcf_assets_dir: str = "assets"
    mjcf_add_actuators: bool = True


def _resolve_per_joint(value: PerJoint | PerJointStr, joint_name: str, what: str):
    if not isinstance(value, dict):
        return value
    matches = [v for pattern, v in value.items() if re.search(pattern, joint_name)]
    if len(matches) > 1:
        raise ValueError(f"Multiple {what} patterns match joint {joint_name!r}")
    return matches[0] if matches else None


def apply_config(robot: Robot, config: ConvertConfig) -> None:
    """Apply generic preprocessing and drive injection to the IR, in place."""
    if config.exclude_geometry_names:
        _exclude_geometry(robot, config.exclude_geometry_names)
    if config.replace_cylinders_with_capsules:
        _replace_cylinders(robot)
    if config.frame_link_mass is not None and config.frame_link_inertia is not None:
        _add_frame_link_inertials(robot, config.frame_link_mass, config.frame_link_inertia)
    if config.skip_collision_inertia or config.skip_collision_mass:
        _skip_collision_mass_properties(
            robot, skip_inertia=config.skip_collision_inertia, skip_mass=config.skip_collision_mass
        )
    if config.joint_drive is not None:
        _inject_drives(robot, config.joint_drive)


def _exclude_geometry(robot: Robot, patterns: list[str]) -> None:
    compiled = [re.compile(p) for p in patterns]

    def keep(geom) -> bool:
        return geom.name is None or not any(p.search(geom.name) for p in compiled)

    for link in robot.links.values():
        removed = [g.name for g in link.visuals + link.collisions if not keep(g)]
        link.visuals = [g for g in link.visuals if keep(g)]
        link.collisions = [g for g in link.collisions if keep(g)]
        for name in removed:
            logger.info("Removed geometry %r from link %r", name, link.name)


def _skip_collision_mass_properties(robot: Robot, *, skip_inertia: bool, skip_mass: bool) -> None:
    for link in robot.links.values():
        if not link.collisions or link.inertial is None:
            continue
        if skip_inertia:
            link.inertial.inertia = None
            link.inertial.origin = None
        if skip_mass:
            link.inertial.mass = None
        if link.inertial.mass is None and link.inertial.inertia is None:
            link.inertial = None  # nothing authored; consumer computes everything
        logger.info(
            "Skipped authored %s on collision link %r",
            "+".join(p for p, on in (("mass", skip_mass), ("inertia", skip_inertia)) if on),
            link.name,
        )


def _replace_cylinders(robot: Robot) -> None:
    for link in robot.links.values():
        for geom in link.visuals + link.collisions:
            if isinstance(geom.shape, Cylinder):
                geom.shape = Capsule(radius=geom.shape.radius, length=geom.shape.length)


def _add_frame_link_inertials(robot: Robot, mass: float, diagonal_inertia: float) -> None:
    # Overwrites any authored inertial: a frame-only link (no visual/collision)
    # has no geometry for the simulator to compute inertia from, so opting into
    # this normalizes all such links to the given negligible inertial.
    for link in frame_only_links(robot):
        link.inertial = Inertial(mass=mass, inertia=np.eye(3) * diagonal_inertia)
        logger.info("Set frame-only link %r inertial to mass=%g", link.name, mass)


def _inject_drives(robot: Robot, cfg: JointDriveCfg) -> None:
    for joint in robot.joints.values():
        if joint.type == "fixed":
            continue
        target_type = _resolve_per_joint(cfg.target_type, joint.name, "target_type")
        drive_type = _resolve_per_joint(cfg.drive_type, joint.name, "drive_type")
        stiffness = _resolve_per_joint(cfg.stiffness, joint.name, "stiffness")
        damping = _resolve_per_joint(cfg.damping, joint.name, "damping")
        joint.drive = JointDrive(
            target_type=target_type if target_type is not None else "position",
            drive_type=drive_type if drive_type is not None else "force",
            stiffness=stiffness if stiffness is not None else 0.0,
            damping=damping if damping is not None else 0.0,
        )
