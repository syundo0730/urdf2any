from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def two_link_urdf() -> Path:
    return FIXTURES / "two_link.urdf"


@pytest.fixture(scope="session")
def mesh_urdf(tmp_path_factory) -> Path:
    """A URDF that references a generated STL mesh, written to a temp dir."""
    out = tmp_path_factory.mktemp("mesh_robot")
    mesh_dir = out / "meshes"
    mesh_dir.mkdir()
    box = trimesh.creation.box(extents=[0.1, 0.1, 0.2])
    box.export(mesh_dir / "link.stl")

    urdf = out / "robot.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="mesh_robot">
  <link name="base_link">
    <inertial><mass value="1.0"/></inertial>
    <visual>
      <geometry><mesh filename="meshes/link.stl"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="meshes/link.stl"/></geometry>
    </collision>
  </link>
  <link name="second">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <collision>
      <geometry><mesh filename="meshes/link.stl" scale="2 2 2"/></geometry>
    </collision>
  </link>
  <joint name="slider" type="prismatic">
    <parent link="base_link"/>
    <child link="second"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0.5" effort="50" velocity="1.0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return urdf
