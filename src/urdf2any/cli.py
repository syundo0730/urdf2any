"""Command-line interface for urdf2any."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConvertConfig, JointDriveCfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urdf2any",
        description="Convert URDF to USD or MJCF (format inferred from output suffix).",
    )
    parser.add_argument("input", type=Path, help="Source URDF file")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file (.usd/.usda or .xml/.mjcf)"
    )
    parser.add_argument(
        "--mesh-root", type=Path, default=None, help="Base dir for package:// mesh URIs"
    )
    parser.add_argument("--dump-ir", action="store_true", help="Print the parsed IR and exit")

    parser.add_argument("--fix-base", action="store_true", help="Weld the root link to the world")
    parser.add_argument("--self-collision", action="store_true")
    parser.add_argument(
        "--collider-type",
        choices=["convex_hull", "convex_decomposition"],
        default="convex_hull",
    )
    parser.add_argument("--replace-cylinders-with-capsules", action="store_true")
    parser.add_argument(
        "--merge-fixed-joints",
        action="store_true",
        help="Collapse fixed-joint links into their parent (combine mass/inertia/geometry)",
    )
    parser.add_argument(
        "--keep-fixed-link-inertials",
        dest="merge_fixed_ignore_inertia",
        action="store_false",
        help="When merging, keep links with mass/inertia as separate bodies "
        "(Isaac Sim 5.1+ behavior); default merges all fixed links",
    )
    parser.add_argument(
        "--compute-missing-inertials",
        action="store_true",
        help="When merging mixes authored and geometry-derived masses, compute the "
        "missing inertial from collision geometry via MuJoCo (requires mujoco)",
    )
    parser.add_argument(
        "--skip-collision-inertia",
        action="store_true",
        help="Omit authored inertia/COM on collision links; let the consumer compute them",
    )
    parser.add_argument(
        "--skip-collision-mass",
        action="store_true",
        help="Omit authored mass on collision links; let the consumer compute it",
    )

    parser.add_argument(
        "--joint-target-type", choices=["position", "velocity", "none"], default="position"
    )
    parser.add_argument("--joint-stiffness", type=float, default=100.0, help="N·m/rad (SI)")
    parser.add_argument("--joint-damping", type=float, default=1.0)

    parser.add_argument(
        "--frame-link-mass",
        type=float,
        default=None,
        help="Add this mass to frame-only links (requires --frame-link-inertia)",
    )
    parser.add_argument("--frame-link-inertia", type=float, default=None)
    parser.add_argument(
        "--exclude-geometry",
        action="append",
        default=[],
        metavar="REGEX",
        help="Drop visual/collision geometry whose name matches (repeatable)",
    )
    parser.add_argument("--no-actuators", action="store_true", help="MJCF: omit <actuator>")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from . import parse_urdf

    if args.dump_ir:
        robot = parse_urdf(args.input, mesh_root=args.mesh_root)
        _dump_ir(robot)
        return 0

    if args.output is None:
        print("error: -o/--output is required (unless --dump-ir)", file=sys.stderr)
        return 2

    config = ConvertConfig(
        fix_base=args.fix_base,
        self_collision=args.self_collision,
        collider_type=args.collider_type,
        replace_cylinders_with_capsules=args.replace_cylinders_with_capsules,
        merge_fixed_joints=args.merge_fixed_joints,
        merge_fixed_ignore_inertia=args.merge_fixed_ignore_inertia,
        compute_missing_inertials=args.compute_missing_inertials,
        skip_collision_inertia=args.skip_collision_inertia,
        skip_collision_mass=args.skip_collision_mass,
        joint_drive=JointDriveCfg(
            target_type=args.joint_target_type,
            stiffness=args.joint_stiffness,
            damping=args.joint_damping,
        ),
        frame_link_mass=args.frame_link_mass,
        frame_link_inertia=args.frame_link_inertia,
        exclude_geometry_names=args.exclude_geometry,
        mjcf_add_actuators=not args.no_actuators,
    )

    from . import convert

    output = convert(args.input, args.output, config=config, mesh_root=args.mesh_root)
    print(f"Wrote {output}")
    return 0


def _dump_ir(robot) -> None:
    print(f"robot: {robot.name}")
    print(f"  root link: {robot.root_link_name()}")
    print(f"  links ({len(robot.links)}):")
    for link in robot.links.values():
        mass = link.inertial.mass if link.inertial else None
        print(
            f"    {link.name}: mass={mass} visuals={len(link.visuals)} "
            f"collisions={len(link.collisions)}"
        )
    print(f"  joints ({len(robot.joints)}):")
    for joint in robot.joints.values():
        print(f"    {joint.name} [{joint.type}] {joint.parent} -> {joint.child}")


if __name__ == "__main__":
    raise SystemExit(main())
