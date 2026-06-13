from __future__ import annotations

import math

import numpy as np
import pytest

from urdf2any import ConvertConfig, JointDriveCfg, convert
from urdf2any.config import apply_config
from urdf2any.urdf.parser import parse_urdf

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def _convert(urdf, tmp_path, **cfg_kwargs):
    out = tmp_path / "robot.usda"
    config = ConvertConfig(**cfg_kwargs)
    convert(urdf, out, config=config)
    return Usd.Stage.Open(str(out))


def test_stage_metadata_and_default_prim(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert stage.GetDefaultPrim().GetName() == "two_link"


def test_rigid_bodies_and_mass(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path)
    base = stage.GetPrimAtPath("/two_link/base_link")
    assert base.HasAPI(UsdPhysics.RigidBodyAPI)
    mass_api = UsdPhysics.MassAPI(base)
    assert mass_api.GetMassAttr().Get() == 2.0
    com = mass_api.GetCenterOfMassAttr().Get()
    assert np.allclose([com[0], com[1], com[2]], [0, 0, 0.05])


def test_link_world_placement(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path)
    arm = UsdGeom.Xformable(stage.GetPrimAtPath("/two_link/arm_link"))
    xform = arm.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = xform.ExtractTranslation()
    assert np.allclose([translation[0], translation[1], translation[2]], [0, 0, 0.1])


def test_revolute_joint_degree_limits(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path)
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/two_link/joints/shoulder"))
    assert joint.GetAxisAttr().Get() == "Y"
    # URDF -1.57 rad -> about -90 deg
    assert math.isclose(joint.GetLowerLimitAttr().Get(), math.degrees(-1.57), rel_tol=1e-5)
    assert joint.GetBody0Rel().GetTargets()[0].pathString == "/two_link/base_link"
    assert joint.GetBody1Rel().GetTargets()[0].pathString == "/two_link/arm_link"


def test_drive_gain_unit_conversion(two_link_urdf, tmp_path):
    stage = _convert(
        two_link_urdf,
        tmp_path,
        joint_drive=JointDriveCfg(stiffness=100.0, damping=2.0, target_type="position"),
    )
    joint = stage.GetPrimAtPath("/two_link/joints/shoulder")
    drive = UsdPhysics.DriveAPI(joint, "angular")
    # SI stiffness 100 N·m/rad -> degrees: 100 * pi/180
    assert math.isclose(drive.GetStiffnessAttr().Get(), 100.0 * math.pi / 180, rel_tol=1e-5)
    assert math.isclose(drive.GetDampingAttr().Get(), 2.0 * math.pi / 180, rel_tol=1e-5)
    assert drive.GetMaxForceAttr().Get() == 10.0


def test_fixed_joint_is_fixed(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path)
    prim = stage.GetPrimAtPath("/two_link/joints/tip_fixed")
    assert prim.IsA(UsdPhysics.FixedJoint)


def test_articulation_root_floating(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path, fix_base=False)
    base = stage.GetPrimAtPath("/two_link/base_link")
    assert base.HasAPI(UsdPhysics.ArticulationRootAPI)
    # PhysX schemas are authored without the plugin; GetAppliedSchemas() filters
    # unregistered schemas, but the typeinfo and authored attributes persist.
    assert "PhysxArticulationAPI" in base.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert base.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 32


def test_articulation_root_fixed_base(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path, fix_base=True)
    root_joint = stage.GetPrimAtPath("/two_link/root_joint")
    assert root_joint.IsA(UsdPhysics.FixedJoint)
    assert root_joint.HasAPI(UsdPhysics.ArticulationRootAPI)


def test_physx_joint_attrs_schemaless(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path)
    prim = stage.GetPrimAtPath("/two_link/joints/shoulder")
    assert "PhysxJointAPI" in prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert math.isclose(prim.GetAttribute("physxJoint:jointFriction").Get(), 0.05, rel_tol=1e-5)


def test_frame_only_link_gets_fallback_inertia(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path)
    tip = stage.GetPrimAtPath("/two_link/tip_frame")
    diag = UsdPhysics.MassAPI(tip).GetDiagonalInertiaAttr().Get()
    assert diag is not None and diag[0] > 0


def test_frame_link_inertial_injection(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path, frame_link_mass=1e-5, frame_link_inertia=4e-8)
    tip = stage.GetPrimAtPath("/two_link/tip_frame")
    assert math.isclose(UsdPhysics.MassAPI(tip).GetMassAttr().Get(), 1e-5, rel_tol=1e-5)


def test_collision_api_on_collider(two_link_urdf, tmp_path):
    stage = _convert(two_link_urdf, tmp_path)
    found = [
        p
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.CollisionAPI) and "collisions" in p.GetPath().pathString
    ]
    assert found


def test_mesh_authoring(mesh_urdf, tmp_path):
    out = tmp_path / "mesh.usda"
    convert(mesh_urdf, out)
    stage = Usd.Stage.Open(str(out))
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert meshes
    mesh = UsdGeom.Mesh(meshes[0])
    assert len(mesh.GetPointsAttr().Get()) > 0
    assert len(mesh.GetFaceVertexIndicesAttr().Get()) > 0


def test_exclude_geometry(two_link_urdf, tmp_path):
    robot = parse_urdf(two_link_urdf)
    # name the cylinder visual then exclude it
    robot.links["arm_link"].visuals[0].name = "arm_visual_mesh"
    apply_config(robot, ConvertConfig(exclude_geometry_names=["arm_visual"]))
    assert not robot.links["arm_link"].visuals
