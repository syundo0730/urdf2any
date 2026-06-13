# urdf2any

[![CI](https://github.com/syundo0730/urdf2any/actions/workflows/ci.yml/badge.svg)](https://github.com/syundo0730/urdf2any/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/urdf2any)](https://pypi.org/project/urdf2any/)

A pure-Python tool that converts URDF to USD / MJCF, with no dependency on
Isaac Sim, Isaac Lab, or Omniverse.

No GPU, no docker, no Omniverse Kit required — it runs on pip packages alone
(`usd-core`, `trimesh`, `numpy`). Two-stage design: URDF → intermediate
representation (IR) → USD / MJCF backends.

## Installation

```bash
pip install urdf2any
# To validate MJCF output by loading it in MuJoCo:
pip install "urdf2any[mjcf-validate]"
```

## Usage

The output format is inferred from the file extension.

```bash
urdf2any robot.urdf -o robot.usd   # USD (OpenUSD UsdPhysics)
urdf2any robot.urdf -o robot.xml   # MJCF (MuJoCo)
urdf2any robot.urdf --dump-ir      # print the parsed IR
```

Python API:

```python
from urdf2any import convert, ConvertConfig, JointDriveCfg

convert(
    "robot.urdf",
    "robot.usd",
    ConvertConfig(joint_drive=JointDriveCfg(stiffness=100.0, damping=1.0)),
)
```

Robot-specific preprocessing (renaming links, removing parts, etc.) is kept out
of the core: obtain the IR with `parse_urdf`, mutate it, and emit with
`convert_robot`.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv build       # produce wheel + sdist in dist/
```

Pushing a `vX.Y.Z` tag publishes to PyPI automatically
(GitHub Actions + PyPI Trusted Publishing, `.github/workflows/release.yml`).

### Parity testing

`tests/test_golden_parity.py` checks that urdf2any's USD output stays physics-equivalent
to the IsaacLab URDF importer for vendored public sample robots (`tests/assets/`). It runs
on CPU in normal CI by diffing against committed golden summaries (`tests/golden/`).

Regenerating the golden requires an IsaacLab container (GPU not needed for the import):

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work \
    nvcr.io/nvidia/isaac-lab:<tag> /isaac-sim/python.sh scripts/generate_reference.py
```

Canonical conversion flags for each sample live in `tests/parity/samples.json` (shared by
the test and the reference generator). Compare two USDs directly with
`python tools/compare_usd.py compare <ref.json|usd> <cand.usd>`.
