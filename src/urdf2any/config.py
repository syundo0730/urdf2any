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
from .transforms import Transform, quat_to_matrix

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
    # Collapse links joined by a fixed joint into their parent (combining mass,
    # inertia, geometry; re-parenting downstream joints). Default off preserves
    # the one-body-per-link output.
    merge_fixed_joints: bool = False
    # When merging, also collapse fixed children that carry mass/inertia.
    # True  -> merge everything (Isaac Sim <=2.4.31 / Isaac Lab default behaviour).
    # False -> keep links with mass/inertia as separate bodies, collapsing only
    #          massless reference frames (Isaac Sim 5.1+ behaviour).
    # Mirrors the importer's set_merge_fixed_ignore_inertia knob.
    merge_fixed_ignore_inertia: bool = True
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
    if config.merge_fixed_joints:
        _merge_fixed_joints(robot, ignore_inertia=config.merge_fixed_ignore_inertia)
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


def _has_body(link) -> bool:
    """Whether a link carries mass/inertia (i.e. is a real body, not a frame)."""
    inertial = link.inertial
    if inertial is None:
        return False
    return bool(inertial.mass and inertial.mass > 0.0) or inertial.inertia is not None


def _merge_fixed_joints(robot: Robot, ignore_inertia: bool = True) -> None:
    """Collapse fixed-joint children into their parent link, in place.

    Port of the URDF importer's ``mergeFixedChildLinks``
    (urdf-importer-extension ImportHelpers.cpp:113). Depth-first so a child has
    already absorbed its own fixed descendants before it is merged upward. The
    merged child link and its fixed joint are removed (unlike the importer, which
    keeps them as frames); downstream joints are re-parented onto the parent.

    ``ignore_inertia`` selects the importer behaviour: True merges every fixed
    child (Isaac Sim <=2.4.31 / Isaac Lab), False keeps children that carry
    mass/inertia as separate bodies and collapses only massless frames
    (Isaac Sim 5.1+).
    """
    # Snapshot the original child-joint lists; mutations below re-parent joints
    # on the live robot, but the traversal follows the original tree.
    original_children = {name: list(joints) for name, joints in robot.child_joints().items()}

    def visit(link_name: str) -> None:
        for joint in original_children[link_name]:
            visit(joint.child)
            if joint.type != "fixed" or joint.name not in robot.joints:
                continue
            if not ignore_inertia and _has_body(robot.links[joint.child]):
                continue  # Isaac Sim 5.1+: keep mass/inertia links as full bodies
            transform = joint.origin
            parent = robot.links[link_name]
            child = robot.links[joint.child]
            parent.inertial = _combine_inertials(parent.inertial, child.inertial, transform)
            for geom in child.visuals:
                geom.origin = transform @ geom.origin
                parent.visuals.append(geom)
            for geom in child.collisions:
                geom.origin = transform @ geom.origin
                parent.collisions.append(geom)
            # Re-parent the child's remaining (downstream) joints onto the parent.
            for other in robot.joints.values():
                if other.parent == joint.child:
                    other.parent = link_name
                    other.origin = transform @ other.origin
            del robot.joints[joint.name]
            del robot.links[joint.child]
            logger.info("Merged fixed-joint link %r into %r", joint.child, link_name)

    visit(robot.root_link_name())


def _combine_inertials(
    parent: Inertial | None, child: Inertial | None, child_to_parent: Transform
) -> Inertial | None:
    """Combine a child link's inertial into the parent's (parallel-axis theorem).

    Both are expressed in the parent link frame, summed about the combined COM,
    and stored in the parent's inertial-origin orientation. Returns the parent
    inertial unchanged if neither link carries mass.
    """
    pm = parent.mass if parent is not None and parent.mass is not None else 0.0
    cm = child.mass if child is not None and child.mass is not None else 0.0
    if pm <= 0.0 and cm <= 0.0:
        return parent

    p_origin = parent.origin if parent is not None and parent.origin is not None else Transform()
    c_origin = child_to_parent @ (
        child.origin if child is not None and child.origin is not None else Transform()
    )
    total_mass = pm + cm
    com = (pm * p_origin.translation + cm * c_origin.translation) / total_mass

    def _tensor_in_parent_frame(inertial, origin, mass):
        if inertial is not None and inertial.inertia is not None:
            local = inertial.inertia
        else:
            local = np.zeros((3, 3))
        rot = quat_to_matrix(origin.rotation)
        delta = origin.translation - com
        parallel = mass * (float(delta @ delta) * np.eye(3) - np.outer(delta, delta))
        return rot @ local @ rot.T + parallel

    summed = _tensor_in_parent_frame(parent, p_origin, pm) + _tensor_in_parent_frame(child, c_origin, cm)
    rot_parent = quat_to_matrix(p_origin.rotation)
    inertia = rot_parent.T @ summed @ rot_parent  # express in the parent's inertial orientation
    return Inertial(
        mass=total_mass,
        origin=Transform(translation=com, rotation=p_origin.rotation),
        inertia=inertia,
    )


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
