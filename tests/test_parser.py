from __future__ import annotations

import numpy as np
import pytest

from urdf2any.ir import Box, Cylinder, MeshRef
from urdf2any.urdf.parser import parse_urdf


def test_parse_two_link(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    assert robot.name == "two_link"
    assert set(robot.links) == {"base_link", "arm_link", "tip_frame"}
    assert set(robot.joints) == {"shoulder", "tip_fixed"}
    assert robot.root_link_name() == "base_link"


def test_parse_inertial(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    base = robot.links["base_link"]
    assert base.inertial.mass == 2.0
    assert base.inertial.inertia[0, 0] == 0.01
    assert np.allclose(base.inertial.origin.translation, [0, 0, 0.05])


def test_frame_only_link_has_no_inertial(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    assert robot.links["tip_frame"].inertial is None
    assert not robot.links["tip_frame"].visuals


def test_parse_joint_limits_and_axis(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    shoulder = robot.joints["shoulder"]
    assert shoulder.type == "revolute"
    assert np.allclose(shoulder.axis, [0, 1, 0])
    assert shoulder.limit.lower == -1.57
    assert shoulder.limit.effort == 10.0
    assert shoulder.dynamics.damping == 0.1
    assert np.allclose(shoulder.origin.translation, [0, 0, 0.1])


def test_geometry_shapes(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    assert isinstance(robot.links["base_link"].visuals[0].shape, Box)
    assert isinstance(robot.links["arm_link"].visuals[0].shape, Cylinder)


def test_world_transforms(two_link_urdf):
    robot = parse_urdf(two_link_urdf)
    world = robot.link_world_transforms()
    assert np.allclose(world["base_link"].translation, [0, 0, 0])
    assert np.allclose(world["arm_link"].translation, [0, 0, 0.1])
    assert np.allclose(world["tip_frame"].translation, [0, 0, 0.4])


def test_mesh_resolution_and_scale(mesh_urdf):
    robot = parse_urdf(mesh_urdf)
    mesh = robot.links["second"].collisions[0].shape
    assert isinstance(mesh, MeshRef)
    assert mesh.path.exists()
    assert np.allclose(mesh.scale, [2, 2, 2])


def test_revolute_without_limit_raises(tmp_path):
    urdf = tmp_path / "bad.urdf"
    urdf.write_text(
        """<robot name="bad">
  <link name="a"/>
  <link name="b"/>
  <joint name="j" type="revolute">
    <parent link="a"/><child link="b"/>
    <axis xyz="1 0 0"/>
  </joint>
</robot>"""
    )
    with pytest.raises(ValueError, match="requires a <limit>"):
        parse_urdf(urdf)


def test_unsupported_joint_type_raises(tmp_path):
    urdf = tmp_path / "bad.urdf"
    urdf.write_text(
        """<robot name="bad">
  <link name="a"/>
  <link name="b"/>
  <joint name="j" type="planar">
    <parent link="a"/><child link="b"/>
  </joint>
</robot>"""
    )
    with pytest.raises(ValueError, match="unsupported type"):
        parse_urdf(urdf)
