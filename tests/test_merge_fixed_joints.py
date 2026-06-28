"""Tests for fixed-joint merging (ConvertConfig.merge_fixed_joints)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from urdf2any import ConvertConfig
from urdf2any.config import apply_config
from urdf2any.urdf.parser import parse_urdf


def _no_drive() -> ConvertConfig:
    return ConvertConfig(merge_fixed_joints=True, joint_drive=None)


def test_two_link_merges_fixed_frame(two_link_urdf):
    # tip_frame is a frame-only link attached to arm_link by a fixed joint.
    robot = parse_urdf(two_link_urdf)
    assert "tip_frame" in robot.links
    apply_config(robot, _no_drive())
    assert "tip_frame" not in robot.links  # merged away
    assert "tip_fixed" not in robot.joints
    assert "shoulder" in robot.joints  # the revolute joint is untouched
    # arm_link kept its own inertia (frame had no mass, contributes nothing)
    assert robot.links["arm_link"].inertial.mass == 0.5


def test_mass_and_com_combine(tmp_path):
    # Two point-like links of equal mass on the x-axis, joined by a fixed joint:
    # combined mass = 2, COM halfway between them.
    urdf = tmp_path / "two.urdf"
    urdf.write_text(
        textwrap.dedent(
            """\
            <robot name="two">
              <link name="parent">
                <inertial><origin xyz="0 0 0"/><mass value="1.0"/>
                  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
              </link>
              <link name="child">
                <inertial><origin xyz="0 0 0"/><mass value="1.0"/>
                  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
              </link>
              <joint name="weld" type="fixed">
                <parent link="parent"/><child link="child"/>
                <origin xyz="2 0 0" rpy="0 0 0"/>
              </joint>
            </robot>
            """
        )
    )
    robot = parse_urdf(urdf)
    apply_config(robot, _no_drive())
    assert set(robot.links) == {"parent"}
    inertial = robot.links["parent"].inertial
    assert inertial.mass == pytest.approx(2.0)
    # COM halfway between (0,0,0) and (2,0,0)
    assert np.allclose(inertial.origin.translation, [1.0, 0.0, 0.0])
    # Parallel-axis: each 1kg mass is 1m from the combined COM along x.
    # Ixx unchanged by x-offset (0.01+0.01); Iyy,Izz gain m*d^2 = 1*1 each side.
    assert inertial.inertia[0, 0] == pytest.approx(0.02)
    assert inertial.inertia[1, 1] == pytest.approx(0.02 + 2.0)
    assert inertial.inertia[2, 2] == pytest.approx(0.02 + 2.0)


def test_downstream_joint_reparented(tmp_path):
    # base -(fixed)-> mid -(revolute)-> tip : merging removes mid and re-parents
    # the revolute onto base, composing the origin.
    urdf = tmp_path / "chain.urdf"
    urdf.write_text(
        textwrap.dedent(
            """\
            <robot name="chain">
              <link name="base"><inertial><mass value="1"/>
                <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
              <link name="mid"><inertial><mass value="1"/>
                <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
              <link name="tip"><inertial><mass value="1"/>
                <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
              <joint name="fix" type="fixed">
                <parent link="base"/><child link="mid"/><origin xyz="1 0 0"/>
              </joint>
              <joint name="hinge" type="revolute">
                <parent link="mid"/><child link="tip"/><origin xyz="0 1 0"/>
                <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="5" velocity="2"/>
              </joint>
            </robot>
            """
        )
    )
    robot = parse_urdf(urdf)
    apply_config(robot, _no_drive())
    assert "mid" not in robot.links
    hinge = robot.joints["hinge"]
    assert hinge.parent == "base"
    # origin composed: base->mid (1,0,0) then mid->tip (0,1,0) == (1,1,0)
    assert np.allclose(hinge.origin.translation, [1.0, 1.0, 0.0])
    assert robot.root_link_name() == "base"


def test_merge_disabled_by_default(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    apply_config(robot, ConvertConfig(joint_drive=None))
    assert "tip_frame" in robot.links  # untouched when merge_fixed_joints=False


def _body_then_frame_urdf(tmp_path) -> Path:
    # base -(fixed)-> mid(mass) -(fixed)-> frame(no mass)
    urdf = tmp_path / "bf.urdf"
    urdf.write_text(
        textwrap.dedent(
            """\
            <robot name="bf">
              <link name="base"><inertial><mass value="1"/>
                <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
              <link name="mid"><inertial><mass value="2"/>
                <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
              <link name="frame"/>
              <joint name="f1" type="fixed"><parent link="base"/><child link="mid"/>
                <origin xyz="0.5 0 0"/></joint>
              <joint name="f2" type="fixed"><parent link="mid"/><child link="frame"/>
                <origin xyz="0.1 0 0"/></joint>
            </robot>
            """
        )
    )
    return urdf


def test_ignore_inertia_true_merges_everything(tmp_path):
    robot = parse_urdf(_body_then_frame_urdf(tmp_path))
    apply_config(robot, ConvertConfig(merge_fixed_joints=True, merge_fixed_ignore_inertia=True,
                                      joint_drive=None))
    assert set(robot.links) == {"base"}  # mid (mass) and frame both folded in
    assert robot.links["base"].inertial.mass == pytest.approx(3.0)


def test_ignore_inertia_false_keeps_mass_bodies(tmp_path):
    robot = parse_urdf(_body_then_frame_urdf(tmp_path))
    apply_config(robot, ConvertConfig(merge_fixed_joints=True, merge_fixed_ignore_inertia=False,
                                      joint_drive=None))
    # mid carries mass -> kept as a separate body; only the massless frame is merged.
    assert set(robot.links) == {"base", "mid"}
    assert "frame" not in robot.links
    assert robot.links["base"].inertial.mass == pytest.approx(1.0)
    assert robot.links["mid"].inertial.mass == pytest.approx(2.0)
