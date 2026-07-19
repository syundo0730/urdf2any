"""Parity check: urdf2any USD output vs the IsaacLab importer golden.

CPU-only, no GPU/Isaac Sim needed: convert the vendored sample URDFs with
urdf2any using the canonical flags (tests/parity/samples.json), extract the same
physics summary, and diff against the committed golden JSON produced by
scripts/generate_reference.py (run separately inside an IsaacLab container).

Golden files are regenerated only when a sample, urdf2any's intended output, or
the Isaac Lab version changes — see scripts/generate_reference.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from urdf2any import ConvertConfig, JointDriveCfg, convert_robot, parse_urdf

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = json.loads((REPO_ROOT / "tests" / "parity" / "samples.json").read_text())["samples"]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

sys.path.insert(0, str(REPO_ROOT / "tools"))
from compare_usd import compare, extract  # noqa: E402

pytest.importorskip("pxr")


def _config(sample: dict) -> ConvertConfig:
    drive = JointDriveCfg(
        target_type=sample["joint_target_type"],
        stiffness=sample["joint_stiffness"],
        damping=sample["joint_damping"],
    )
    if sample["merge_fixed_joints"]:
        # The merged samples validate the isaaclab compatibility preset
        # end-to-end against the real importer's default (merging) behaviour.
        return ConvertConfig.isaaclab(
            fix_base=sample["fix_base"],
            collider_type=sample["collider_type"],
            skip_collision_inertia=sample["skip_collision_inertia"],
            joint_drive=drive,
        )
    return ConvertConfig(
        fix_base=sample["fix_base"],
        collider_type=sample["collider_type"],
        skip_collision_inertia=sample["skip_collision_inertia"],
        joint_drive=drive,
    )


@pytest.mark.parametrize("name", list(SAMPLES))
def test_golden_parity(name, tmp_path):
    golden_file = GOLDEN_DIR / f"{name}.json"
    if not golden_file.exists():
        pytest.skip(f"golden {golden_file.name} not generated yet")

    sample = SAMPLES[name]
    robot = parse_urdf(REPO_ROOT / sample["urdf"])
    out = convert_robot(robot, tmp_path / f"{name}.usda", _config(sample))

    candidate = extract(str(out))
    golden = json.loads(golden_file.read_text())["summary"]

    diffs = compare(golden, candidate)
    assert not diffs, "urdf2any output diverged from IsaacLab golden:\n" + "\n".join(diffs)
