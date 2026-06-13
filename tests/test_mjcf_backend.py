from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from urdf2any import ConvertConfig, JointDriveCfg, convert


def _convert(urdf, tmp_path, name="robot.xml", **cfg_kwargs):
    out = tmp_path / name
    convert(urdf, out, config=ConvertConfig(**cfg_kwargs))
    return out, ET.parse(out).getroot()


def test_compiler_uses_radians(two_link_urdf, tmp_path):
    _, root = _convert(two_link_urdf, tmp_path)
    compiler = root.find("compiler")
    assert compiler.get("angle") == "radian"


def test_body_nesting_follows_kinematics(two_link_urdf, tmp_path):
    _, root = _convert(two_link_urdf, tmp_path)
    base = root.find(".//body[@name='base_link']")
    assert base is not None
    arm = base.find("body[@name='arm_link']")
    assert arm is not None
    tip = arm.find("body[@name='tip_frame']")
    assert tip is not None  # fixed joint -> nested body, no joint element


def test_fixed_joint_emits_no_joint(two_link_urdf, tmp_path):
    _, root = _convert(two_link_urdf, tmp_path)
    tip = root.find(".//body[@name='tip_frame']")
    assert tip.find("joint") is None


def test_revolute_joint_radian_range(two_link_urdf, tmp_path):
    _, root = _convert(two_link_urdf, tmp_path)
    joint = root.find(".//joint[@name='shoulder']")
    assert joint.get("type") == "hinge"
    lower, upper = (float(v) for v in joint.get("range").split())
    assert np.isclose(lower, -1.57) and np.isclose(upper, 1.57)
    assert np.isclose(float(joint.get("damping")), 0.1)


def test_freejoint_when_floating(two_link_urdf, tmp_path):
    _, root = _convert(two_link_urdf, tmp_path, fix_base=False)
    base = root.find(".//body[@name='base_link']")
    assert base.find("freejoint") is not None


def test_no_freejoint_when_fixed(two_link_urdf, tmp_path):
    _, root = _convert(two_link_urdf, tmp_path, fix_base=True)
    base = root.find(".//body[@name='base_link']")
    assert base.find("freejoint") is None


def test_inertial_fullinertia(two_link_urdf, tmp_path):
    _, root = _convert(two_link_urdf, tmp_path)
    base = root.find(".//body[@name='base_link']")
    inertial = base.find("inertial")
    assert inertial is not None
    assert np.isclose(float(inertial.get("mass")), 2.0)
    full = [float(v) for v in inertial.get("fullinertia").split()]
    # ixx iyy izz ixy ixz iyz
    assert np.allclose(full, [0.01, 0.01, 0.005, 0, 0, 0])


def test_box_size_is_half_extent(two_link_urdf, tmp_path):
    _, root = _convert(two_link_urdf, tmp_path)
    geoms = root.findall(".//body[@name='base_link']/geom[@type='box']")
    assert geoms
    size = [float(v) for v in geoms[0].get("size").split()]
    assert np.allclose(size, [0.1, 0.1, 0.05])


def test_actuator_generated(two_link_urdf, tmp_path):
    _, root = _convert(
        two_link_urdf, tmp_path, joint_drive=JointDriveCfg(stiffness=50.0, target_type="position")
    )
    actuator = root.find("actuator")
    assert actuator is not None
    pos = actuator.find("position[@joint='shoulder']")
    assert np.isclose(float(pos.get("kp")), 50.0)
    assert np.allclose([float(v) for v in pos.get("forcerange").split()], [-10, 10])


def test_no_actuators_flag(two_link_urdf, tmp_path):
    _, root = _convert(two_link_urdf, tmp_path, mjcf_add_actuators=False)
    assert root.find("actuator") is None


def test_mesh_assets_copied(mesh_urdf, tmp_path):
    out, root = _convert(mesh_urdf, tmp_path)
    meshes = root.findall(".//asset/mesh")
    assert meshes
    assets_dir = out.parent / "assets"
    for mesh in meshes:
        assert (assets_dir / mesh.get("file")).exists()
    # scaled mesh keeps its scale attribute
    assert any(m.get("scale") for m in meshes)


def _world_link_urdf(tmp_path):
    urdf = tmp_path / "with_world.urdf"
    urdf.write_text(
        """<robot name="with_world">
  <link name="world">
    <inertial><mass value="0.001"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>
  </link>
  <link name="arm">
    <inertial><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="j" type="revolute">
    <parent link="world"/><child link="arm"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="5" velocity="2"/>
  </joint>
</robot>"""
    )
    return urdf


def test_reserved_world_link_renamed(tmp_path):
    urdf = _world_link_urdf(tmp_path)
    out, root = _convert(urdf, tmp_path, fix_base=True)
    body_names = [b.get("name") for b in root.iter("body")]
    # MuJoCo reserves "world"; the link must be emitted under a different name.
    assert "world" not in body_names
    assert "world_link" in body_names


# --- validation against the real MuJoCo compiler ---------------------------

mujoco = pytest.importorskip("mujoco")


def test_reserved_world_link_loads_in_mujoco(tmp_path):
    urdf = _world_link_urdf(tmp_path)
    out, _ = _convert(urdf, tmp_path, fix_base=True)
    model = mujoco.MjModel.from_xml_path(str(out))  # must not raise "repeated name 'world'"
    assert model.njnt == 1


def test_two_link_loads_in_mujoco(two_link_urdf, tmp_path):
    out, _ = _convert(two_link_urdf, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(out))
    assert model.njnt >= 1
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)  # one step must not raise


def test_mesh_robot_loads_in_mujoco(mesh_urdf, tmp_path):
    out, _ = _convert(mesh_urdf, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(out))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)


def test_actuated_model_steps(two_link_urdf, tmp_path):
    out, _ = _convert(two_link_urdf, tmp_path, fix_base=True)
    model = mujoco.MjModel.from_xml_path(str(out))
    assert model.nu == 1  # one actuator for the shoulder
    data = mujoco.MjData(model)
    data.ctrl[0] = 0.5
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
