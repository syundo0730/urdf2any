"""MJCF parity: urdf2any output vs MuJoCo's native URDF conversion.

Both are compiled by MuJoCo and compared as MjModels (see tools/compare_mjcf).
CPU-only and live (no golden file): the reference is MuJoCo loading the same
URDF. Uses merge_fixed_joints=True so the body sets align with MuJoCo, which
always merges fixed joints on URDF import, and compute_missing_inertials=True so
geometry-derived masses survive merging into authored-inertial links.

Known reference-side quirk (verified): MuJoCo's native URDF
importer misplaces the COM of inertials merged through fixed-joint chains that
involve rotations/offsets (franka panda_link7). urdf2any's composition matches
MuJoCo's own welded-body composition exactly, so only the *inertia tensor* of
such links is excluded from the comparison — masses must still match.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from urdf2any import ConvertConfig, JointDriveCfg, convert_robot, parse_urdf

mujoco = pytest.importorskip("mujoco")

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = json.loads((REPO_ROOT / "tests" / "parity" / "samples.json").read_text())["samples"]

sys.path.insert(0, str(REPO_ROOT / "tools"))
from compare_mjcf import compare, extract  # noqa: E402


def _models(sample_name: str, tmp_path: Path):
    """Return (mujoco_native_ref, urdf2any_cand) summaries for a sample."""
    urdf = REPO_ROOT / SAMPLES[sample_name]["urdf"]
    # The mujoco preset IS the compatibility claim under test here; fix_base
    # matches MuJoCo treating a fixed/"world" base as the static anchor.
    cfg = ConvertConfig.mujoco(
        fix_base=True,
        joint_drive=JointDriveCfg(target_type="position", stiffness=100, damping=1),
    )
    out = convert_robot(parse_urdf(urdf), tmp_path / f"{sample_name}.xml", cfg)
    cand = extract(mujoco.MjModel.from_xml_path(str(out)))
    ref = extract(mujoco.MjModel.from_xml_path(str(urdf)))
    return ref, cand


def test_simple_2_link_full_parity(tmp_path):
    # Authored inertials throughout -> kinematics AND dynamic-body mass/inertia match.
    ref, cand = _models("simple_2_link", tmp_path)
    diffs = compare(ref, cand, compare_masses=True)
    assert not diffs, "MJCF diverged from MuJoCo-native:\n" + "\n".join(diffs)


def test_franka_kinematic_parity(tmp_path):
    ref, cand = _models("franka", tmp_path)
    diffs = compare(ref, cand, compare_masses=False)
    assert not diffs, "MJCF diverged from MuJoCo-native:\n" + "\n".join(diffs)
    assert ref["nv"] == cand["nv"] == 9  # sanity: the 9 DOF survived merging on both sides


def test_franka_all_dynamic_masses_match(tmp_path):
    # With compute_missing_inertials, every dynamic body's MASS matches MuJoCo.
    # The only tolerated diff is panda_link7's inertia tensor, where the native
    # importer's own merged-COM placement is wrong (verified against MuJoCo's
    # welded-body composition, which urdf2any matches to 5 decimals).
    ref, cand = _models("franka", tmp_path)
    diffs = compare(ref, cand, compare_masses=True)
    mass_diffs = [d for d in diffs if ": mass" in d]
    assert not mass_diffs, "mass diverged from MuJoCo-native:\n" + "\n".join(mass_diffs)
    others = {d.split()[1].rstrip(":") for d in diffs}
    assert others <= {"panda_link7"}, f"unexpected diffs: {diffs}"
