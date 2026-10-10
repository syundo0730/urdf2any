"""Command-line interface for urdf2any."""

from __future__ import annotations

import argparse
import logging
import sys
import xml.etree.ElementTree as ET
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

    parser.add_argument(
        "--preset",
        choices=["isaaclab", "mujoco"],
        default=None,
        help="Start from a compatibility preset: 'isaaclab' matches the IsaacLab "
        "URDF importer defaults, 'mujoco' matches MuJoCo's native URDF conversion. "
        "Individual flags below override the preset.",
    )
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
    parser.add_argument(
        "--skin", type=Path, help="MJCF: attach a triangulated skinned DAE (requires urdf2any[skin])"
    )
    parser.add_argument(
        "--skin-bone", action="append", default=[], type=_bone_link, metavar="BONE=LINK",
        help="MJCF: map a Collada joint sid to a URDF link (repeat for each weighted bone)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _bone_link(value: str) -> tuple[str, str]:
    bone, separator, link = value.partition("=")
    if not separator or not bone.strip() or not link.strip() or "=" in link:
        raise argparse.ArgumentTypeError("expected BONE=LINK with non-empty names")
    return bone.strip(), link.strip()


def _attach_skin(robot, output: Path, config: ConvertConfig, path: Path, bone_links: dict[str, str]) -> None:
    """Attach skins using the preprocessed IR and the backend's emitted body names.

    Skin files resolve through meshdir; texture paths are model-relative because
    the backend leaves texturedir unset. XML asset names avoid existing names,
    and meshes sharing a texture reuse one texture asset.
    """
    from . import load_skinned_meshes, mjcf_body_names, write_mjcf_skins

    skins = load_skinned_meshes(path, robot, bone_links)
    skin_assets = write_mjcf_skins(
        skins, output.parent / config.mjcf_assets_dir,
        name_prefix="skin", body_names=mjcf_body_names(robot, config),
    )
    tree = ET.parse(output)
    asset_element = tree.getroot().find("asset")
    if asset_element is None:
        asset_element = ET.SubElement(tree.getroot(), "asset")
    names = {element.get("name") for element in asset_element}

    def unique_name(base: str) -> str:
        name = base
        suffix = 1
        while name in names:
            name = f"{base}_{suffix}"
            suffix += 1
        names.add(name)
        return name

    textures: dict[str, str] = {}
    for skin in skin_assets:
        attributes = {"name": unique_name(skin.name), "file": skin.file}
        if skin.texture_file is not None:
            texture_name = textures.get(skin.texture_file)
            if texture_name is None:
                texture_name = unique_name(f"{skin.name}_texture")
                ET.SubElement(
                    asset_element, "texture", name=texture_name, type="2d",
                    file=(Path(config.mjcf_assets_dir) / skin.texture_file).as_posix(),
                )
                textures[skin.texture_file] = texture_name
            material_name = unique_name(f"{skin.name}_material")
            ET.SubElement(asset_element, "material", name=material_name, texture=texture_name)
            attributes["material"] = material_name
        else:
            attributes["rgba"] = " ".join(str(value) for value in skin.rgba)
        ET.SubElement(asset_element, "skin", **attributes)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def _build_config(args) -> ConvertConfig:
    """Build a ConvertConfig from CLI args, layering: preset -> explicit flags.

    A flag overrides the preset only when it was actually passed (its value
    differs from the argparse default), so ``--preset mujoco`` keeps its merge
    settings unless the user explicitly changes them.
    """
    presets = {"isaaclab": ConvertConfig.isaaclab, "mujoco": ConvertConfig.mujoco}
    config = presets[args.preset]() if args.preset else ConvertConfig()

    parser = _build_parser()
    flag_fields = (
        "fix_base",
        "self_collision",
        "collider_type",
        "replace_cylinders_with_capsules",
        "merge_fixed_joints",
        "merge_fixed_ignore_inertia",
        "compute_missing_inertials",
        "skip_collision_inertia",
        "skip_collision_mass",
        "frame_link_mass",
        "frame_link_inertia",
    )
    for name in flag_fields:
        value = getattr(args, name)
        if value != parser.get_default(name):
            setattr(config, name, value)

    config.joint_drive = JointDriveCfg(
        target_type=args.joint_target_type,
        stiffness=args.joint_stiffness,
        damping=args.joint_damping,
    )
    config.exclude_geometry_names = args.exclude_geometry
    config.mjcf_add_actuators = not args.no_actuators
    return config


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.skin_bone and args.skin is None:
        parser.error("--skin-bone requires --skin")
    if args.skin is not None:
        if args.dump_ir or args.output is None or args.output.suffix.lower() not in (".xml", ".mjcf"):
            parser.error("--skin requires MJCF output (-o model.xml or model.mjcf) and cannot use --dump-ir")
        if not args.skin_bone:
            parser.error("--skin requires --skin-bone BONE=LINK for each weighted bone")
    bone_links = {}
    for bone, link in args.skin_bone:
        if bone in bone_links:
            parser.error(f"duplicate --skin-bone mapping for {bone!r}")
        bone_links[bone] = link
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

    config = _build_config(args)

    from . import convert, convert_robot

    if args.skin is None:
        output = convert(args.input, args.output, config=config, mesh_root=args.mesh_root)
    else:
        try:
            robot = parse_urdf(args.input, mesh_root=args.mesh_root)
            output = convert_robot(robot, args.output, config=config)
            _attach_skin(robot, output, config, args.skin, bone_links)
        except (ImportError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
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
