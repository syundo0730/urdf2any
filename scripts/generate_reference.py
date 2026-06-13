"""Generate golden parity summaries from the IsaacLab URDF importer.

This is the *reference* side of the parity check. It must run inside an Isaac Sim
runtime (the IsaacLab importer needs Kit + an RTX GPU), e.g.:

    # canonical (released image):
    docker run --rm --gpus all -v "$PWD:/work" -w /work \\
        nvcr.io/nvidia/isaac-lab:<tag> /isaac-sim/python.sh scripts/generate_reference.py

    # or inside an existing IsaacLab container with the repo mounted:
    /isaac-sim/python.sh scripts/generate_reference.py

For each sample in tests/parity/samples.json it runs IsaacLab's UrdfConverter with
the canonical flags, then writes tests/golden/<sample>.json (the physics summary
extracted by tools/compare_usd.extract). Commit those golden files; everyday CI
(tests/test_golden_parity.py) diffs urdf2any's output against them on CPU.

merge_fixed_joints is taken from samples.json (false) to match urdf2any; note that
IsaacLab's UrdfConverterCfg defaults it to True.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_JSON = REPO_ROOT / "tests" / "parity" / "samples.json"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", nargs="*", default=None, help="subset of sample names")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Imports that require the running Kit app.
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tools"))
from compare_usd import extract  # noqa: E402


def _isaac_version() -> str:
    try:
        from isaaclab.utils.version import get_isaac_sim_version

        return str(get_isaac_sim_version())
    except Exception:
        return "unknown"


def _convert(sample: dict, out_dir: Path) -> str:
    cfg = UrdfConverterCfg(
        asset_path=str(REPO_ROOT / sample["urdf"]),
        usd_dir=str(out_dir),
        usd_file_name="reference.usd",
        fix_base=sample["fix_base"],
        merge_fixed_joints=sample["merge_fixed_joints"],
        collider_type=sample["collider_type"],
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type=sample["joint_target_type"],
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=sample["joint_stiffness"],
                damping=sample["joint_damping"],
            ),
        ),
    )
    return UrdfConverter(cfg).usd_path


def main() -> int:
    config = json.loads(SAMPLES_JSON.read_text())["samples"]
    names = args.samples or list(config)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    isaac_version = _isaac_version()

    for name in names:
        sample = config[name]
        with tempfile.TemporaryDirectory() as tmp:
            usd_path = _convert(sample, Path(tmp))
            summary = extract(usd_path)
        golden = {
            "_meta": {
                "sample": name,
                "isaac_sim_version": isaac_version,
                "generated": datetime.datetime.now().isoformat(timespec="seconds"),
                "config": sample,
            },
            "summary": summary,
        }
        out = GOLDEN_DIR / f"{name}.json"
        out.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
        print(
            f"[{name}] wrote {out.relative_to(REPO_ROOT)}: "
            f"{len(summary['links'])} links, {len(summary['joints'])} joints, "
            f"roots={summary['articulation_roots']}"
        )
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        simulation_app.close()
    sys.exit(rc)
