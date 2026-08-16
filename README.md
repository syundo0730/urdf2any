# urdf2any

[![CI](https://github.com/syundo0730/urdf2any/actions/workflows/ci.yml/badge.svg)](https://github.com/syundo0730/urdf2any/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/urdf2any)](https://pypi.org/project/urdf2any/)
[![Python](https://img.shields.io/pypi/pyversions/urdf2any)](https://pypi.org/project/urdf2any/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Convert URDF robot descriptions to **USD** (OpenUSD / Isaac Sim) and **MJCF**
(MuJoCo) — in pure Python, with no Isaac Sim, Omniverse, or GPU required.

- **Self-contained** — runs anywhere `pip` works (`usd-core`, `trimesh`, `numpy`)
- **One source of truth** — a single simulator-agnostic intermediate
  representation drives both backends, so USD and MJCF describe the same
  physical robot
- **Verified** — output is checked in CI against the real reference
  implementations: the IsaacLab URDF importer (golden comparison) and MuJoCo's
  native URDF loader (live comparison)

## Installation

```bash
pip install urdf2any
```

The base install covers URDF → USD / MJCF conversion. Optional extras unlock
features that need an extra dependency:

| Install | Adds | Unlocks |
|---|---|---|
| `pip install urdf2any` | — | All conversion features: USD/MJCF output, mesh geometry, fixed-joint merging, drives, materials |
| `pip install "urdf2any[mjcf-validate]"` | `mujoco` | Geometry-derived inertials (`--compute-missing-inertials`, needed by `--preset mujoco`) |
| `pip install "urdf2any[skin]"` | `pycollada` | Skinned Collada (DAE) → MuJoCo `<skin>` conversion (`urdf2any.load_skinned_meshes`) |

Extras combine: `pip install "urdf2any[mjcf-validate,skin]"`.

## Quick start

The output format is inferred from the file extension:

```bash
urdf2any robot.urdf -o robot.usd   # USD (UsdPhysics articulation)
urdf2any robot.urdf -o robot.xml   # MJCF (with actuators)
```

Or from Python:

```python
from urdf2any import convert, ConvertConfig, JointDriveCfg

convert(
    "robot.urdf",
    "robot.usd",
    ConvertConfig(joint_drive=JointDriveCfg(stiffness=100.0, damping=1.0)),
)
```

By default every URDF link becomes one body and no lossy transformation is
applied — the same config yields the same robot in both formats.

## Compatibility presets

To reproduce the behavior of an existing converter, start from a preset;
individual flags still override it:

```bash
urdf2any robot.urdf -o robot.usd --preset isaaclab   # IsaacLab URDF importer defaults
urdf2any robot.urdf -o robot.xml --preset mujoco     # MuJoCo native URDF conversion
```

| Preset | Matches | Effect |
|---|---|---|
| `isaaclab` | IsaacLab `UrdfConverter` defaults | merges all fixed-joint links |
| `mujoco` | MuJoCo's built-in URDF loader | merges fixed-joint links and derives missing link masses from collision geometry (needs `pip install "urdf2any[mjcf-validate]"`) |

The same presets are available in Python as `ConvertConfig.isaaclab()` /
`ConvertConfig.mujoco()`, and their definitions are what the parity test suites
validate against the real implementations.

## Options

Conversion flags correspond to `ConvertConfig` fields, mostly under the same
name in snake_case. The exceptions:

- `--keep-fixed-link-inertials` sets `merge_fixed_ignore_inertia=False`
- `--exclude-geometry` patterns go in the `exclude_geometry_names` list
- `--no-actuators` sets `mjcf_add_actuators=False`
- the `--joint-*` flags fill the nested drive config:
  `ConvertConfig(joint_drive=JointDriveCfg(target_type=..., stiffness=..., damping=...))`
- the General table below is CLI-only; in Python, `mesh_root` is an argument of
  `convert` / `parse_urdf`, not a config field

### General

| Option | Default | Description |
|---|---|---|
| `input` | required | Source URDF file |
| `-o`, `--output` | — | Output file; the format is inferred from the suffix (`.usd`/`.usda` → USD, `.xml`/`.mjcf` → MJCF) |
| `--mesh-root PATH` | — | Base directory for resolving `package://` mesh URIs |
| `--preset {isaaclab,mujoco}` | — | Start from a [compatibility preset](#compatibility-presets); individual flags override it |
| `--dump-ir` | off | Print the parsed intermediate representation and exit (no output file) |
| `-v`, `--verbose` | off | Log conversion details at INFO level |

### Physics & structure

| Option | Default | Description |
|---|---|---|
| `--fix-base` | off | Weld the root link to the world (fixed-base robot) |
| `--self-collision` | off | Enable collisions between the robot's own bodies |
| `--collider-type {convex_hull,convex_decomposition}` | `convex_hull` | Collision approximation for mesh geometry |
| `--replace-cylinders-with-capsules` | off | Convert cylinder collision shapes to capsules |
| `--merge-fixed-joints` | off | Collapse fixed-joint links into their parent, composing mass/inertia/geometry |
| `--keep-fixed-link-inertials` | off (merge all) | When merging, keep fixed links that have mass/inertia as separate bodies (Isaac Sim 5.1+ behavior) |
| `--compute-missing-inertials` | off | When merging mixes authored and missing masses, derive the missing inertial from collision geometry via MuJoCo (needs `urdf2any[mjcf-validate]`) |
| `--skip-collision-inertia` | off | Omit authored inertia/COM on links with collision geometry; the consumer computes them |
| `--skip-collision-mass` | off | Omit authored mass on links with collision geometry; the consumer computes it |
| `--frame-link-mass KG` | — | Set this mass on frame-only links (no visual or collision geometry), overwriting any authored inertial; requires `--frame-link-inertia` |
| `--frame-link-inertia KGM2` | — | Diagonal inertia to pair with `--frame-link-mass` |
| `--exclude-geometry REGEX` | — | Drop visual/collision geometry whose name matches; repeatable |

### Joint drives

| Option | Default | Description |
|---|---|---|
| `--joint-target-type {position,velocity,none}` | `position` | Drive target: position/velocity PD, or `none` for passive joints |
| `--joint-stiffness VALUE` | `100.0` | Drive stiffness in SI: N·m/rad (revolute) or N/m (prismatic) |
| `--joint-damping VALUE` | `1.0` | Drive damping in SI units |

### MJCF only

| Option | Default | Description |
|---|---|---|
| `--no-actuators` | off (actuators on) | Omit the `<actuator>` section from the MJCF output |

## Feature support

| | |
|---|---|
| Joints | revolute, continuous, prismatic, fixed (optional merging with inertia composition) |
| Geometry | box, sphere, cylinder, capsule, mesh (STL / OBJ / DAE via trimesh) |
| Inertials | authored mass/inertia/COM; optional geometry-derived backfill via MuJoCo |
| Drives | position / velocity PD gains → USD DriveAPI / MJCF actuators |
| Materials | URDF colors → UsdPreviewSurface / MJCF materials |
| Skins | skinned Collada (DAE) → MuJoCo `<skin>` assets, bone weights rebound to link frames (`pip install "urdf2any[skin]"`) |
| Not yet | mimic joints, sensors, UsdSkel skins, `package://` ROS package resolution beyond a search root |

Robot-specific preprocessing (renaming links, removing parts) is deliberately
out of scope: parse with `parse_urdf`, mutate the IR, then emit with
`convert_robot`.

## Development

```bash
uv sync
uv run pytest        # includes the parity suites (CPU only)
uv run ruff check .
```

Releases are published to PyPI by pushing a `vX.Y.Z` tag
(Trusted Publishing via `.github/workflows/release.yml`).

### Parity testing

- **USD vs IsaacLab importer** — `tests/test_golden_parity.py` diffs urdf2any's
  output against committed golden summaries (`tests/golden/`) generated by the
  real importer. Regenerate them inside an IsaacLab container with
  `scripts/generate_reference.py` (the import itself needs no GPU).
- **MJCF vs MuJoCo** — `tests/test_mjcf_parity.py` compiles both urdf2any's MJCF
  and the source URDF with MuJoCo and compares the resulting models live; no
  golden files needed.

Sample robots under `tests/assets/` are vendored from
[IsaacLab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause) and
[Isaac Sim](https://github.com/isaac-sim/IsaacSim) (Apache-2.0); see the
`NOTICE` file next to each.

## License

[Apache-2.0](LICENSE)
