"""Geometry-derived inertial computation, delegated to MuJoCo.

Builds a throwaway single-body MJCF containing a link's collision geoms and lets
MuJoCo's compiler derive mass / COM / inertia (default density 1000 kg/m3, same
mesh-inertia conventions as MuJoCo's native URDF import). This avoids
re-implementing mesh inertia math and guarantees consistency with MuJoCo.

Used by fixed-joint merging (``ConvertConfig.compute_missing_inertials``): when a
geometry-massed link is merged into a link with an authored inertial, the
authored value would otherwise mask the geometry and its mass would be lost.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .ir import Box, Capsule, Cylinder, Inertial, Link, MeshRef, Sphere
from .transforms import Transform


def _fmt(values) -> str:
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    return " ".join(f"{v:.9g}" for v in arr)


def compute_link_inertial(link: Link) -> Inertial | None:
    """Compute the collision-geometry inertial of a link via MuJoCo.

    Returns None when the link has no collision geometry. Raises ImportError if
    the optional ``mujoco`` dependency is missing.
    """
    if not link.collisions:
        return None
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_missing_inertials requires the 'mujoco' package "
            "(pip install 'urdf2any[mjcf-validate]')"
        ) from exc

    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="probe")
    ET.SubElement(body, "freejoint")

    for i, geometry in enumerate(link.collisions):
        shape = geometry.shape
        attrs: dict[str, str] = {"name": f"g{i}"}
        if np.any(geometry.origin.translation):
            attrs["pos"] = _fmt(geometry.origin.translation)
        if not np.allclose(geometry.origin.rotation, [1.0, 0.0, 0.0, 0.0]):
            attrs["quat"] = _fmt(geometry.origin.rotation)
        if isinstance(shape, Box):
            attrs["type"] = "box"
            attrs["size"] = _fmt(shape.size / 2.0)
        elif isinstance(shape, Sphere):
            attrs["type"] = "sphere"
            attrs["size"] = _fmt(shape.radius)
        elif isinstance(shape, Cylinder):
            attrs["type"] = "cylinder"
            attrs["size"] = _fmt([shape.radius, shape.length / 2.0])
        elif isinstance(shape, Capsule):
            attrs["type"] = "capsule"
            attrs["size"] = _fmt([shape.radius, shape.length / 2.0])
        elif isinstance(shape, MeshRef):
            mesh_attrs = {"name": f"m{i}", "file": str(shape.path)}
            if not np.allclose(shape.scale, 1.0):
                mesh_attrs["scale"] = _fmt(shape.scale)
            ET.SubElement(asset, "mesh", **mesh_attrs)
            attrs["type"] = "mesh"
            attrs["mesh"] = f"m{i}"
        else:  # pragma: no cover
            raise TypeError(f"Unsupported shape: {shape}")
        ET.SubElement(body, "geom", **attrs)

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    b = model.body("probe")
    mass = float(b.mass[0])
    if mass <= 0.0:
        return None
    # MuJoCo reports principal moments (b.inertia) in the inertial frame b.iquat
    # (w,x,y,z — same convention as the IR) located at b.ipos: exactly the
    # URDF-style Inertial(origin frame + tensor about COM).
    return Inertial(
        mass=mass,
        origin=Transform(
            translation=np.array(b.ipos, dtype=float),
            rotation=np.array(b.iquat, dtype=float),
        ),
        inertia=np.diag(np.array(b.inertia, dtype=float)),
    )
