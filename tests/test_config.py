from __future__ import annotations

from urdf2any import ConvertConfig, convert_robot
from urdf2any.config import apply_config
from urdf2any.urdf.parser import parse_urdf


def test_skip_collision_inertia_clears_tensor_keeps_mass(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    # base_link has collision + authored inertia
    assert robot.links["base_link"].inertial.inertia is not None
    apply_config(robot, ConvertConfig(skip_collision_inertia=True, joint_drive=None))
    base = robot.links["base_link"]
    assert base.inertial.mass == 2.0  # mass kept
    assert base.inertial.inertia is None  # tensor cleared
    assert base.inertial.origin is None  # COM cleared


def test_skip_collision_mass_clears_mass_keeps_inertia(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    apply_config(robot, ConvertConfig(skip_collision_mass=True, joint_drive=None))
    base = robot.links["base_link"]
    assert base.inertial.mass is None  # mass cleared
    assert base.inertial.inertia is not None  # inertia kept


def test_skip_both_drops_inertial_entirely(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    apply_config(
        robot,
        ConvertConfig(skip_collision_inertia=True, skip_collision_mass=True, joint_drive=None),
    )
    assert robot.links["base_link"].inertial is None


def test_skip_collision_inertia_keeps_frame_link_inertia(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    apply_config(
        robot,
        ConvertConfig(
            skip_collision_inertia=True,
            frame_link_mass=1e-5,
            frame_link_inertia=4e-8,
            joint_drive=None,
        ),
    )
    # frame-only link still gets its injected inertia (no collision -> not stripped)
    tip = robot.links["tip_frame"]
    assert tip.inertial is not None
    assert tip.inertial.inertia is not None


def test_convert_robot_ir_entry_point(two_link_urdf, tmp_path):
    robot = parse_urdf(two_link_urdf)
    robot.name = "renamed"
    out = convert_robot(robot, tmp_path / "r.usda", ConvertConfig())
    assert out.exists()
    from pxr import Usd

    stage = Usd.Stage.Open(str(out))
    assert stage.GetDefaultPrim().GetName() == "renamed"
