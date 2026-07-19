# urdf2any

A pure-Python tool that converts URDF to USD / MJCF, independent of Isaac Sim and Omniverse.

## Architecture

URDF → parser (`src/urdf2any/urdf/parser.py`) → simulator-agnostic IR
(`src/urdf2any/ir.py`) → backends (`src/urdf2any/backends/usd.py`,
`src/urdf2any/backends/mjcf.py`). Generic IR preprocessing (fixed-joint merging,
drive injection, geometry exclusion) lives in `src/urdf2any/config.py`.

## Rules

- The IR uses SI units and radians. Unit conversion is the backend's
  responsibility (USD uses degrees).
- Keep robot-specific policy (link name literals, part removal, gain values) out
  of the core; provide only generic, declarative mechanisms. Callers mutate the
  IR via `parse_urdf` → `convert_robot`.
- Long-lived knowledge belongs in code, not documents: module docstrings state
  conventions and known divergences; the parity suites encode verified behavior.
  Do not add documentation files beyond the README.
- Compatibility claims must be backed by a parity test. After changing
  conversion behavior, run `uv run pytest` — `tests/test_golden_parity.py`
  compares against IsaacLab importer goldens (`tests/golden/`) and
  `tests/test_mjcf_parity.py` compares live against MuJoCo's native URDF loader.
- Regenerating USD goldens requires an IsaacLab container:
  `scripts/generate_reference.py` (see its docstring; the import needs no GPU).
