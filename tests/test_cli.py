"""End-to-end tests of the ``urdf2any`` command-line interface.

These run the CLI as a real subprocess (not by calling main() in-process) so the
argument parsing, dispatch, file output, and the installed console-script entry
point are all exercised the way a user invokes them.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

# Invoke via the module so the test never depends on PATH; a separate test below
# covers the installed `urdf2any` console script.
CLI = [sys.executable, "-m", "urdf2any.cli"]


def _run(args, **kwargs):
    return subprocess.run(CLI + args, capture_output=True, text=True, **kwargs)


def test_cli_to_usd(two_link_urdf, tmp_path):
    out = tmp_path / "robot.usd"
    result = _run([str(two_link_urdf), "-o", str(out)])
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert "Wrote" in result.stdout

    from pxr import Usd

    stage = Usd.Stage.Open(str(out))
    assert stage.GetDefaultPrim().GetName() == "two_link"


def test_cli_to_mjcf(two_link_urdf, tmp_path):
    out = tmp_path / "robot.xml"
    result = _run([str(two_link_urdf), "-o", str(out)])
    assert result.returncode == 0, result.stderr
    root = ET.parse(out).getroot()
    assert root.tag == "mujoco"


def test_cli_dump_ir(two_link_urdf):
    result = _run([str(two_link_urdf), "--dump-ir"])
    assert result.returncode == 0, result.stderr
    assert "robot: two_link" in result.stdout
    assert "base_link" in result.stdout


def test_cli_options_flow_through(two_link_urdf, tmp_path):
    out = tmp_path / "robot.usda"
    result = _run([str(two_link_urdf), "-o", str(out), "--fix-base", "--joint-stiffness", "50"])
    assert result.returncode == 0, result.stderr

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(out))
    # --fix-base authors a fixed root_joint carrying the articulation root.
    root_joint = stage.GetPrimAtPath("/two_link/root_joint")
    assert root_joint.IsValid()
    assert root_joint.HasAPI(UsdPhysics.ArticulationRootAPI)


def test_cli_requires_output(two_link_urdf):
    result = _run([str(two_link_urdf)])  # neither -o nor --dump-ir
    assert result.returncode == 2
    assert "output" in result.stderr.lower()


def test_cli_unknown_suffix(two_link_urdf, tmp_path):
    result = _run([str(two_link_urdf), "-o", str(tmp_path / "robot.foo")])
    assert result.returncode != 0


def test_cli_preset_mujoco_merges(two_link_urdf, tmp_path):
    out = tmp_path / "robot.xml"
    result = _run([str(two_link_urdf), "-o", str(out), "--preset", "mujoco", "--fix-base"])
    assert result.returncode == 0, result.stderr
    body_names = [b.get("name") for b in ET.parse(out).getroot().iter("body")]
    # the mujoco preset merges fixed joints: the fixed tip_frame link is gone
    assert "tip_frame" not in body_names


def test_cli_flag_overrides_preset(two_link_urdf, tmp_path):
    out = tmp_path / "robot.xml"
    # preset turns merging on; --keep-fixed-link-inertials flips ignore_inertia off,
    # which keeps mass-carrying links but still folds massless frames like tip_frame.
    result = _run(
        [str(two_link_urdf), "-o", str(out), "--preset", "isaaclab", "--keep-fixed-link-inertials"]
    )
    assert result.returncode == 0, result.stderr
    body_names = [b.get("name") for b in ET.parse(out).getroot().iter("body")]
    assert "arm_link" in body_names  # mass-carrying link kept as its own body
    assert "tip_frame" not in body_names  # massless frame still merged


def test_console_script_entry_point(two_link_urdf, tmp_path):
    script = shutil.which("urdf2any")
    if script is None:
        pytest.skip("urdf2any console script not on PATH")
    out = tmp_path / "robot.usd"
    result = subprocess.run(
        [script, str(two_link_urdf), "-o", str(out)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
