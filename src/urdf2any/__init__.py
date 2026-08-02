"""urdf2any: convert URDF robot descriptions to USD and MJCF.

This is a general-purpose, simulator-agnostic tool. Robot-specific policy
(link renaming, part removal, gain choices) belongs in caller code; this
package provides only the mechanisms.
"""

from __future__ import annotations

from pathlib import Path

from .backends.mjcf import mjcf_body_names
from .backends.mjcf_skin import SkinAsset, write_mjcf_skins
from .config import ConvertConfig, JointDriveCfg
from .ir import Robot
from .skin import SkinnedMesh, SkinnedModel, load_skinned_meshes
from .urdf.parser import parse_urdf

__all__ = [
    "ConvertConfig",
    "JointDriveCfg",
    "Robot",
    "SkinAsset",
    "SkinnedMesh",
    "SkinnedModel",
    "parse_urdf",
    "convert",
    "convert_robot",
    "load_skinned_meshes",
    "mjcf_body_names",
    "write_mjcf_skins",
]


def convert(
    urdf_path: str | Path,
    output_path: str | Path,
    config: ConvertConfig | None = None,
    mesh_root: str | Path | None = None,
) -> Path:
    """Convert a URDF file to USD or MJCF, inferred from ``output_path`` suffix.

    Args:
        urdf_path: Source URDF file.
        output_path: Output file. ``.usd``/``.usda``/``.usdc`` -> USD;
            ``.xml``/``.mjcf`` -> MJCF.
        config: Conversion configuration (drive gains, preprocessing, ...).
        mesh_root: Base directory for resolving ``package://`` mesh URIs.

    Returns:
        The path that was written.
    """
    config = config or ConvertConfig()
    robot = parse_urdf(urdf_path, mesh_root=mesh_root)
    return convert_robot(robot, output_path, config)


def convert_robot(
    robot: Robot,
    output_path: str | Path,
    config: ConvertConfig | None = None,
) -> Path:
    """Convert an in-memory :class:`Robot` IR to USD or MJCF.

    This is the entry point for callers that build or mutate the IR themselves
    (e.g. a project-specific wrapper that renames links before conversion).
    Applies generic preprocessing from ``config`` (drives, frame-link inertials,
    geometry exclusion, ...) then dispatches on the output suffix.
    """
    from .config import apply_config

    config = config or ConvertConfig()
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    apply_config(robot, config)

    if suffix in (".usd", ".usda", ".usdc"):
        from .backends.usd import convert_usd

        return convert_usd(robot, output_path, config)
    if suffix in (".xml", ".mjcf"):
        from .backends.mjcf import convert_mjcf

        return convert_mjcf(robot, output_path, config)
    raise ValueError(f"Cannot infer output format from suffix {suffix!r} (use .usd or .xml)")
