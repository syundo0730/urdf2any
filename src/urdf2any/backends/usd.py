"""URDF IR -> USD conversion using only pip ``usd-core``.

Output follows the prim structure of Isaac Sim's URDF importer
(``isaacsim.asset.importer.urdf``) in a single layer, so the result loads as a
PhysX articulation in Isaac Sim / Isaac Lab. NVIDIA-proprietary PhysX schemas
are authored schema-lessly (``AddAppliedSchema`` + raw attributes), which is
byte-compatible with the plugin-authored output.

Unit conventions (differ from the SI/radian IR):
- rotational joint limits, drive gains, and velocity limits are in degrees
- everything else is SI (stage metersPerUnit = 1)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade

from ..config import ConvertConfig
from ..ir import (
    Box,
    Capsule,
    Cylinder,
    Geometry,
    Joint,
    Link,
    MeshRef,
    Robot,
    Sphere,
)
from ..mesh import SubMesh, load_submeshes
from ..transforms import Transform, quat_align_x_to, quat_multiply

logger = logging.getLogger(__name__)

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0
# Matches kNegligibleMass * 10 in the Isaac Sim importer (UrdfImporter.cpp).
FALLBACK_DIAGONAL_INERTIA = 1.0e-4

_AXIS_TOKENS = {0: "X", 1: "Y", 2: "Z"}


def convert_usd(robot: Robot, output_path: str | Path, config: ConvertConfig) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    writer = _UsdWriter(stage, robot, config)
    writer.write()
    stage.Save()
    logger.info("Wrote %s", output_path)
    return output_path


def _valid(name: str) -> str:
    return Tf.MakeValidIdentifier(name)


def _vec3(v: np.ndarray) -> Gf.Vec3f:
    return Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))


def _quatf(q: np.ndarray) -> Gf.Quatf:
    return Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3])))


def _set_xform(prim, transform: Transform, scale: np.ndarray | None = None) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    t = transform.translation
    q = transform.rotation
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(t[0]), float(t[1]), float(t[2]))
    )
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])))
    )
    s = scale if scale is not None else np.ones(3)
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(s[0]), float(s[1]), float(s[2]))
    )


def _joint_axis_token(axis: np.ndarray) -> tuple[str, np.ndarray]:
    """Pick a USD axis token and a frame-correction quaternion for a URDF axis.

    Mirrors the importer: axes aligned with +/-X/Y/Z use that token (negative
    direction handled by a 180-degree flip); arbitrary axes fall back to "X"
    with a general alignment rotation baked into the joint frames.
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    for i in range(3):
        unit = np.zeros(3)
        unit[i] = 1.0
        d = float(np.dot(a, unit))
        if abs(d) > 1.0 - 1e-6:
            if d > 0:
                return _AXIS_TOKENS[i], np.array([1.0, 0.0, 0.0, 0.0])
            # 180-degree rotation about an axis orthogonal to the joint axis
            ortho = np.zeros(3)
            ortho[(i + 1) % 3] = 1.0
            return _AXIS_TOKENS[i], np.array([0.0, *ortho])
    return "X", quat_align_x_to(a)


class _UsdWriter:
    def __init__(self, stage: Usd.Stage, robot: Robot, config: ConvertConfig):
        self.stage = stage
        self.robot = robot
        self.config = config
        self.root_path = Sdf.Path(f"/{_valid(robot.name)}")
        self.world = robot.link_world_transforms()
        self.material_paths: dict[str, Sdf.Path] = {}
        self._mesh_material_count = 0

    def write(self) -> None:
        robot_prim = UsdGeom.Xform.Define(self.stage, self.root_path)
        _set_xform(robot_prim.GetPrim(), Transform.identity())
        self.stage.SetDefaultPrim(robot_prim.GetPrim())

        self.stage.DefinePrim(self.root_path.AppendChild("Looks"), "Scope")
        for name in self.robot.materials:
            self._material_path(name)

        for link in self.robot.links.values():
            self._write_link(link)

        movable = [j for j in self.robot.joints.values() if j.type != "fixed"]
        if self.robot.joints:
            self.stage.DefinePrim(self.root_path.AppendChild("joints"), "Scope")
        for joint in self.robot.joints.values():
            self._write_joint(joint)

        self._write_articulation_root(has_movable_joints=bool(movable))

    # ----- links ------------------------------------------------------------

    def _link_path(self, link_name: str) -> Sdf.Path:
        return self.root_path.AppendChild(_valid(link_name))

    def _write_link(self, link: Link) -> None:
        prim = UsdGeom.Xform.Define(self.stage, self._link_path(link.name)).GetPrim()
        _set_xform(prim, self.world[link.name])

        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        inertial = link.inertial
        if inertial is not None:
            if inertial.mass is not None:
                mass_api.CreateMassAttr().Set(float(inertial.mass))
            if inertial.inertia is not None:
                from ..transforms import diagonalize_inertia

                diag, principal = diagonalize_inertia(inertial.inertia)
                if inertial.origin is not None:
                    principal = quat_multiply(inertial.origin.rotation, principal)
                mass_api.CreateDiagonalInertiaAttr().Set(_vec3(diag))
                mass_api.CreatePrincipalAxesAttr().Set(_quatf(principal))
            if inertial.origin is not None:
                mass_api.CreateCenterOfMassAttr().Set(_vec3(inertial.origin.translation))
        has_inertia = inertial is not None and inertial.inertia is not None
        if not link.collisions and not has_inertia:
            logger.warning(
                "Link %r has no colliders and no inertia; assigning a small isotropic inertia",
                link.name,
            )
            mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(FALLBACK_DIAGONAL_INERTIA))

        if link.visuals:
            visuals_path = self._link_path(link.name).AppendChild("visuals")
            UsdGeom.Xform.Define(self.stage, visuals_path)
            for i, geometry in enumerate(link.visuals):
                self._write_geometry(visuals_path, geometry, i, collision=False)
        if link.collisions:
            collisions_path = self._link_path(link.name).AppendChild("collisions")
            collisions_xform = UsdGeom.Xform.Define(self.stage, collisions_path)
            # Render as guide geometry so viewers hide colliders by default.
            UsdGeom.Imageable(collisions_xform.GetPrim()).CreatePurposeAttr().Set(
                UsdGeom.Tokens.guide
            )
            for i, geometry in enumerate(link.collisions):
                self._write_geometry(collisions_path, geometry, i, collision=True)

    # ----- geometry ---------------------------------------------------------

    def _write_geometry(
        self, parent_path: Sdf.Path, geometry: Geometry, index: int, collision: bool
    ) -> None:
        shape = geometry.shape
        base = geometry.name or f"{type(shape).__name__.lower()}_{index}"
        path = parent_path.AppendChild(_valid(base))

        if isinstance(shape, MeshRef):
            self._write_mesh(path, geometry, shape, collision)
            return

        scale = None
        if isinstance(shape, Box):
            gprim = UsdGeom.Cube.Define(self.stage, path)
            gprim.CreateSizeAttr().Set(1.0)
            half = Gf.Vec3f(0.5)
            gprim.CreateExtentAttr().Set([-half, half])
            scale = shape.size
        elif isinstance(shape, Sphere):
            gprim = UsdGeom.Sphere.Define(self.stage, path)
            gprim.CreateRadiusAttr().Set(shape.radius)
            r = Gf.Vec3f(shape.radius)
            gprim.CreateExtentAttr().Set([-r, r])
        elif isinstance(shape, Cylinder):
            gprim = UsdGeom.Cylinder.Define(self.stage, path)
            gprim.CreateRadiusAttr().Set(shape.radius)
            gprim.CreateHeightAttr().Set(shape.length)
            gprim.CreateAxisAttr().Set(UsdGeom.Tokens.z)
            e = Gf.Vec3f(shape.radius, shape.radius, shape.length / 2)
            gprim.CreateExtentAttr().Set([-e, e])
        elif isinstance(shape, Capsule):
            gprim = UsdGeom.Capsule.Define(self.stage, path)
            gprim.CreateRadiusAttr().Set(shape.radius)
            gprim.CreateHeightAttr().Set(shape.length)
            gprim.CreateAxisAttr().Set(UsdGeom.Tokens.z)
            e = Gf.Vec3f(shape.radius, shape.radius, shape.length / 2 + shape.radius)
            gprim.CreateExtentAttr().Set([-e, e])
        else:  # pragma: no cover
            raise TypeError(f"Unsupported shape: {shape}")

        _set_xform(gprim.GetPrim(), geometry.origin, scale)
        if collision:
            UsdPhysics.CollisionAPI.Apply(gprim.GetPrim())
        elif geometry.material:
            self._bind_material(gprim.GetPrim(), self._material_path(geometry.material))

    def _write_mesh(
        self, path: Sdf.Path, geometry: Geometry, ref: MeshRef, collision: bool
    ) -> None:
        container = UsdGeom.Xform.Define(self.stage, path)
        _set_xform(container.GetPrim(), geometry.origin, ref.scale)

        for submesh in load_submeshes(ref.path):
            mesh_path = path.AppendChild(_valid(submesh.name))
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            mesh.CreatePointsAttr().Set([Gf.Vec3f(*map(float, v)) for v in submesh.vertices])
            mesh.CreateFaceVertexCountsAttr().Set([3] * len(submesh.faces))
            mesh.CreateFaceVertexIndicesAttr().Set([int(i) for i in submesh.faces.flatten()])
            mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            lo = submesh.vertices.min(axis=0)
            hi = submesh.vertices.max(axis=0)
            mesh.CreateExtentAttr().Set([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
            if submesh.normals is not None:
                mesh.CreateNormalsAttr().Set([Gf.Vec3f(*map(float, n)) for n in submesh.normals])
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            if submesh.uv is not None:
                primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
                )
                primvar.Set([Gf.Vec2f(float(u), float(v)) for u, v in submesh.uv])

            if collision:
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
                approximation = (
                    UsdPhysics.Tokens.convexDecomposition
                    if self.config.collider_type == "convex_decomposition"
                    else UsdPhysics.Tokens.convexHull
                )
                mesh_collision.CreateApproximationAttr().Set(approximation)
            else:
                material_path = None
                if geometry.material:
                    material_path = self._material_path(geometry.material)
                elif submesh.color is not None:
                    material_path = self._adhoc_material(submesh)
                if material_path is not None:
                    self._bind_material(mesh.GetPrim(), material_path)

    # ----- materials ----------------------------------------------------------

    def _material_path(self, name: str) -> Sdf.Path:
        if name in self.material_paths:
            return self.material_paths[name]
        material = self.robot.materials.get(name)
        color = material.color if material is not None and material.color is not None else None
        path = self._define_preview_material(f"material_{name}", color)
        self.material_paths[name] = path
        return path

    def _adhoc_material(self, submesh: SubMesh) -> Sdf.Path:
        self._mesh_material_count += 1
        return self._define_preview_material(
            f"mesh_material_{self._mesh_material_count}", submesh.color
        )

    def _define_preview_material(self, name: str, color: np.ndarray | None) -> Sdf.Path:
        path = self.root_path.AppendChild("Looks").AppendChild(_valid(name))
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path.AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        if color is not None:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
            )
            if len(color) > 3 and color[3] < 1.0:
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(color[3]))
        surface = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(surface)
        return path

    def _bind_material(self, prim, material_path: Sdf.Path) -> None:
        material = UsdShade.Material(self.stage.GetPrimAtPath(material_path))
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    # ----- joints -------------------------------------------------------------

    def _write_joint(self, joint: Joint) -> None:
        path = self.root_path.AppendChild("joints").AppendChild(_valid(joint.name))

        if joint.type == "fixed":
            prim = UsdPhysics.FixedJoint.Define(self.stage, path)
        elif joint.type in ("revolute", "continuous"):
            prim = UsdPhysics.RevoluteJoint.Define(self.stage, path)
        elif joint.type == "prismatic":
            prim = UsdPhysics.PrismaticJoint.Define(self.stage, path)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported joint type {joint.type}")

        prim.CreateBody0Rel().SetTargets([self._link_path(joint.parent)])
        prim.CreateBody1Rel().SetTargets([self._link_path(joint.child)])

        axis_token, axis_quat = (
            _joint_axis_token(joint.axis) if joint.type != "fixed" else ("X", np.array([1.0, 0, 0, 0]))
        )
        local_rot0 = quat_multiply(joint.origin.rotation, axis_quat)
        prim.CreateLocalPos0Attr().Set(_vec3(joint.origin.translation))
        prim.CreateLocalRot0Attr().Set(_quatf(local_rot0))
        prim.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        prim.CreateLocalRot1Attr().Set(_quatf(axis_quat))

        if joint.type == "fixed":
            return

        prim.CreateAxisAttr().Set(axis_token)
        angular = joint.type in ("revolute", "continuous")
        limit_scale = RAD2DEG if angular else 1.0

        if joint.mimic is not None:
            logger.warning("Joint %r mimics %r; mimic is not supported yet", joint.name, joint.mimic.joint)

        if joint.limit is not None and joint.type != "continuous":
            prim.CreateLowerLimitAttr().Set(float(joint.limit.lower) * limit_scale)
            prim.CreateUpperLimitAttr().Set(float(joint.limit.upper) * limit_scale)

        usd_prim = prim.GetPrim()
        # PhysX joint attributes, authored without the PhysxSchema plugin.
        usd_prim.AddAppliedSchema("PhysxJointAPI")
        if joint.dynamics is not None and joint.dynamics.friction > 0:
            usd_prim.CreateAttribute(
                "physxJoint:jointFriction", Sdf.ValueTypeNames.Float, custom=False
            ).Set(float(joint.dynamics.friction))
        max_velocity = (
            float(joint.limit.velocity) * limit_scale
            if joint.limit is not None and joint.limit.velocity > 0
            else 3.4e38
        )
        usd_prim.CreateAttribute(
            "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float, custom=False
        ).Set(max_velocity)

        if joint.type != "continuous":  # importer applies joint state only to limited joints
            state_token = "angular" if angular else "linear"
            usd_prim.AddAppliedSchema(f"PhysxJointStateAPI:{state_token}")
            for attr in ("position", "velocity"):
                usd_prim.CreateAttribute(
                    f"state:{state_token}:physics:{attr}", Sdf.ValueTypeNames.Float, custom=False
                ).Set(0.0)

        self._write_drive(prim, joint, angular)

    def _write_drive(self, joint_prim, joint: Joint, angular: bool) -> None:
        drive = joint.drive
        if drive is None:
            return
        token = "angular" if angular else "linear"
        gain_scale = DEG2RAD if angular else 1.0  # N·m/rad -> N·m/deg
        drive_api = UsdPhysics.DriveAPI.Apply(joint_prim.GetPrim(), token)
        if joint.limit is not None and joint.limit.effort > 0:
            drive_api.CreateMaxForceAttr().Set(float(joint.limit.effort))
        drive_api.CreateTypeAttr().Set(
            UsdPhysics.Tokens.acceleration if drive.drive_type == "acceleration" else UsdPhysics.Tokens.force
        )
        if drive.target_type == "position":
            drive_api.CreateTargetPositionAttr().Set(0.0)
            drive_api.CreateStiffnessAttr().Set(float(drive.stiffness) * gain_scale)
            drive_api.CreateDampingAttr().Set(float(drive.damping) * gain_scale)
        elif drive.target_type == "velocity":
            drive_api.CreateTargetVelocityAttr().Set(0.0)
            drive_api.CreateStiffnessAttr().Set(0.0)
            drive_api.CreateDampingAttr().Set(float(drive.stiffness) * gain_scale)
        else:  # "none": passive joint; expose URDF dynamics as drive damping
            damping = joint.dynamics.damping if joint.dynamics is not None else 0.0
            drive_api.CreateStiffnessAttr().Set(0.0)
            drive_api.CreateDampingAttr().Set(float(damping) * gain_scale)

    # ----- articulation -------------------------------------------------------

    def _write_articulation_root(self, has_movable_joints: bool) -> None:
        root_link = self.robot.root_link_name()
        if self.config.fix_base:
            path = self.root_path.AppendChild("root_joint")
            fixed = UsdPhysics.FixedJoint.Define(self.stage, path)
            fixed.CreateBody1Rel().SetTargets([self._link_path(root_link)])
            target = fixed.GetPrim()
        else:
            if not has_movable_joints:
                return
            target = self.stage.GetPrimAtPath(self._link_path(root_link))

        UsdPhysics.ArticulationRootAPI.Apply(target)
        target.AddAppliedSchema("PhysxArticulationAPI")
        target.CreateAttribute(
            "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, custom=False
        ).Set(self.config.self_collision)
        target.CreateAttribute(
            "physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int, custom=False
        ).Set(32)
        target.CreateAttribute(
            "physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, custom=False
        ).Set(1)
