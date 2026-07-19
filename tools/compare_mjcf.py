#!/usr/bin/env python3
"""Compare a urdf2any MJCF against MuJoCo's own native URDF conversion.

Both the urdf2any ``.xml`` and the source ``.urdf`` are compiled by MuJoCo into
an ``MjModel``; comparing the compiled models (keyed by body/joint name) cancels
out XML formatting and decomposition conventions, leaving only real differences
in how each tool translated the URDF.

Cheap and CPU-only (``pip install mujoco``), so it can run live in CI without a
golden file — the reference is generated on the fly from the URDF.

Known divergences by design:
- The static base/world is represented differently (MuJoCo folds a welded root
  into the worldbody and drops its mass); such bodies are not compared.
- When a link's mass is geometry-derived (no authored <inertial>) and a
  geom-mass child is merged into a parent that *does* have an authored inertial,
  urdf2any keeps the authored value while MuJoCo recomputes from geometry. Mass
  parity therefore holds for authored-inertial models; geometry-derived models
  are compared on kinematics/DOF only.
"""

from __future__ import annotations

import mujoco
import numpy as np

_JOINT_TYPE = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def extract(model: mujoco.MjModel) -> dict:
    """Name-keyed physics summary of a compiled MjModel."""
    bodies = {}
    for i in range(model.nbody):
        b = model.body(i)
        bodies[b.name] = {
            "mass": float(b.mass[0]),
            "principal_inertia": sorted(float(x) for x in b.inertia),
            "has_dof": int(b.jntnum[0]) > 0,
        }
    joints = {}
    for i in range(model.njnt):
        j = model.joint(i)
        joints[j.name] = {
            "type": _JOINT_TYPE[int(j.type[0])],
            "axis": [round(float(x), 6) for x in j.axis],
            "range": [round(float(x), 6) for x in j.range],
        }
    return {"bodies": bodies, "joints": joints, "nv": int(model.nv), "nq": int(model.nq)}


def _close(a, b, rtol=1e-3, atol=1e-6) -> bool:
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))


def compare(ref: dict, cand: dict, *, compare_masses: bool = True, rtol: float = 1e-3) -> list[str]:
    """Differences between two MjModel summaries (empty == equivalent).

    Always compares DOF counts and the moving-joint set (type/axis/range). Body
    mass/inertia is compared only for dynamic bodies present in both models, and
    only when ``compare_masses`` is set (off for geometry-derived-mass models).
    """
    diffs: list[str] = []
    if ref["nv"] != cand["nv"] or ref["nq"] != cand["nq"]:
        diffs.append(f"dof: ref nv={ref['nv']} nq={ref['nq']} cand nv={cand['nv']} nq={cand['nq']}")

    rj, cj = ref["joints"], cand["joints"]
    for n in sorted(set(rj) ^ set(cj)):
        diffs.append(f"joint only in {'reference' if n in rj else 'candidate'}: {n}")
    for n in sorted(set(rj) & set(cj)):
        if rj[n]["type"] != cj[n]["type"]:
            diffs.append(f"joint {n}: type ref={rj[n]['type']} cand={cj[n]['type']}")
        if not _close(rj[n]["axis"], cj[n]["axis"]):
            diffs.append(f"joint {n}: axis ref={rj[n]['axis']} cand={cj[n]['axis']}")
        if not _close(rj[n]["range"], cj[n]["range"]):
            diffs.append(f"joint {n}: range ref={rj[n]['range']} cand={cj[n]['range']}")

    if compare_masses:
        rb, cb = ref["bodies"], cand["bodies"]
        for n in sorted(set(rb) & set(cb)):
            # Skip static bodies (no DOF on either side): the welded base/world
            # is represented differently and is dynamically irrelevant.
            if not rb[n]["has_dof"] and not cb[n]["has_dof"]:
                continue
            if not _close(rb[n]["mass"], cb[n]["mass"], rtol=rtol):
                diffs.append(f"body {n}: mass ref={rb[n]['mass']:.6g} cand={cb[n]['mass']:.6g}")
            elif not _close(rb[n]["principal_inertia"], cb[n]["principal_inertia"], rtol=1e-2):
                diffs.append(
                    f"body {n}: inertia ref={rb[n]['principal_inertia']} "
                    f"cand={cb[n]['principal_inertia']}"
                )
    return diffs


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mjcf", help="urdf2any-produced MJCF (.xml)")
    parser.add_argument("urdf", help="source URDF (compiled by MuJoCo as reference)")
    parser.add_argument("--no-masses", action="store_true", help="compare kinematics/DOF only")
    args = parser.parse_args(argv)

    cand = extract(mujoco.MjModel.from_xml_path(args.mjcf))
    ref = extract(mujoco.MjModel.from_xml_path(args.urdf))
    diffs = compare(ref, cand, compare_masses=not args.no_masses)
    print(f"reference(URDF): nv={ref['nv']} njoints={len(ref['joints'])} nbody={len(ref['bodies'])}")
    print(f"candidate(MJCF): nv={cand['nv']} njoints={len(cand['joints'])} nbody={len(cand['bodies'])}")
    if not diffs:
        print("\nMATCH: no physics-relevant differences within tolerance.")
        return 0
    print(f"\n{len(diffs)} difference(s):")
    for d in diffs:
        print(f"  - {d}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
