#!/usr/bin/env python3
"""Extract and compare physics-relevant content of robot USD files.

Used for golden parity comparison against Isaac Sim's URDF importer output. Prim
*paths* and exact float values differ between authoring tools, so this compares
the physics-meaningful content keyed by link/joint *name* (the leaf of the prim
path), with tolerances, rather than doing a byte/usdcat diff.

``extract`` works both on a self-contained USD (plain ``usd-core``, the candidate)
and on the importer's sublayered/PhysX-schema USD (run inside an Isaac Sim runtime
whose ``pxr`` resolves the sublayers — the reference).

CLI:
    python tools/compare_usd.py extract <usd> [-o summary.json]
    python tools/compare_usd.py compare <reference.json|usd> <candidate.json|usd> [--rtol 1e-3]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from pxr import Usd, UsdPhysics


def _vec(attr) -> list[float] | None:
    if attr and attr.HasAuthoredValue():
        return [float(x) for x in attr.Get()]
    return None


def _quat(attr) -> list[float] | None:
    """Read a quaternion attribute as [w, x, y, z]; None if unauthored."""
    if attr and attr.HasAuthoredValue():
        q = attr.Get()
        im = q.GetImaginary()
        return [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]
    return None


def extract(usd_path: str) -> dict:
    """Extract a physics-relevant summary of a robot USD, keyed by prim name."""
    stage = Usd.Stage.Open(usd_path)
    links: dict[str, dict] = {}
    joints: dict[str, dict] = {}
    articulation_roots: list[str] = []

    for prim in stage.Traverse():
        name = prim.GetName()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # HasAuthoredValue: MassAPI makes these valid schema attributes that
            # return a fallback (mass 0, inertia (0,0,0)) even when unauthored.
            mass_attr = prim.GetAttribute("physics:mass")
            diag = prim.GetAttribute("physics:diagonalInertia")
            com = prim.GetAttribute("physics:centerOfMass")
            pa = prim.GetAttribute("physics:principalAxes")
            has_mass = bool(mass_attr and mass_attr.HasAuthoredValue())
            links[name] = {
                "mass": float(mass_attr.Get()) if has_mass else None,
                "diagonal_inertia": _vec(diag),  # None when unauthored
                "principal_axes": _quat(pa),  # None == identity
                "center_of_mass": _vec(com),
            }
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(name)
        jtype = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            jtype = "revolute"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            jtype = "prismatic"
        elif prim.IsA(UsdPhysics.FixedJoint):
            jtype = "fixed"
        if jtype is not None:
            entry: dict = {"type": jtype}
            if jtype != "fixed":
                axis = prim.GetAttribute("physics:axis")
                lo = prim.GetAttribute("physics:lowerLimit")
                hi = prim.GetAttribute("physics:upperLimit")
                entry["axis"] = axis.Get() if axis else None
                entry["lower"] = float(lo.Get()) if lo and lo.Get() is not None else None
                entry["upper"] = float(hi.Get()) if hi and hi.Get() is not None else None
                for token in ("angular", "linear"):
                    drive = UsdPhysics.DriveAPI(prim, token)
                    s = drive.GetStiffnessAttr()
                    if s and s.HasAuthoredValue():
                        entry["drive_stiffness"] = float(s.Get())
                        d = drive.GetDampingAttr()
                        entry["drive_damping"] = (
                            float(d.Get()) if d and d.HasAuthoredValue() else None
                        )
            joints[name] = entry
    return {"links": links, "joints": joints, "articulation_roots": sorted(articulation_roots)}


# Absolute floor so near-zero values (e.g. off-diagonal inertia components that
# are ~1e-11 floating noise after tensor reconstruction) compare equal to 0.
_ATOL = 1e-9


def _close(a, b, rtol: float) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, rtol) for x, y in zip(a, b, strict=False))
    if a is None or b is None or isinstance(a, str):
        return a == b
    if a == b:  # exact, also covers inf == inf
        return True
    return abs(a - b) <= _ATOL + rtol * max(abs(a), abs(b))


def _quat_matrix(q: list[float] | None) -> np.ndarray:
    if q is None:
        return np.eye(3)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _inertia_tensor(link: dict) -> list[float] | None:
    """Reconstruct the body-frame inertia I = R·diag·Rᵀ from the authored
    (diagonalInertia, principalAxes), so the comparison is independent of the
    diagonalization convention (eigenvalue order / axis assignment)."""
    diag = link.get("diagonal_inertia")
    if diag is None:
        return None
    r = _quat_matrix(link.get("principal_axes"))
    tensor = r @ np.diag(diag) @ r.T
    iu = np.triu_indices(3)
    return [float(v) for v in tensor[iu]]  # 6 unique components


def _norm_mass(m):
    # The importer zeros the welded fixed-base root mass; urdf2any leaves it
    # unauthored. Both mean "no dynamically-meaningful mass" -> treat as equal.
    return None if (m is None or m == 0.0) else m


_JOINT_FIELDS = ("type", "axis", "lower", "upper", "drive_stiffness", "drive_damping")


def compare(ref: dict, cand: dict, rtol: float = 1e-3) -> list[str]:
    """Return a list of human-readable physics differences (empty == MATCH)."""
    diffs: list[str] = []
    if ref["articulation_roots"] != cand["articulation_roots"]:
        diffs.append(
            f"articulation_roots: ref={ref['articulation_roots']} cand={cand['articulation_roots']}"
        )

    rl, cl = ref["links"], cand["links"]
    for nm in sorted(set(rl) - set(cl)):
        diffs.append(f"link only in reference: {nm}")
    for nm in sorted(set(cl) - set(rl)):
        diffs.append(f"link only in candidate: {nm}")
    for nm in sorted(set(rl) & set(cl)):
        if not _close(_norm_mass(rl[nm].get("mass")), _norm_mass(cl[nm].get("mass")), rtol):
            diffs.append(f"link {nm}: mass ref={rl[nm].get('mass')} cand={cl[nm].get('mass')}")
        if not _close(rl[nm].get("center_of_mass"), cl[nm].get("center_of_mass"), rtol):
            diffs.append(
                f"link {nm}: center_of_mass ref={rl[nm].get('center_of_mass')} "
                f"cand={cl[nm].get('center_of_mass')}"
            )
        if not _close(_inertia_tensor(rl[nm]), _inertia_tensor(cl[nm]), rtol):
            diffs.append(
                f"link {nm}: inertia ref(diag={rl[nm].get('diagonal_inertia')}) "
                f"cand(diag={cl[nm].get('diagonal_inertia')})"
            )

    rj, cj = ref["joints"], cand["joints"]
    for nm in sorted(set(rj) - set(cj)):
        diffs.append(f"joint only in reference: {nm}")
    for nm in sorted(set(cj) - set(rj)):
        diffs.append(f"joint only in candidate: {nm}")
    for nm in sorted(set(rj) & set(cj)):
        for f in _JOINT_FIELDS:
            if f in rj[nm] or f in cj[nm]:
                if not _close(rj[nm].get(f), cj[nm].get(f), rtol):
                    diffs.append(f"joint {nm}: {f} ref={rj[nm].get(f)} cand={cj[nm].get(f)}")
    return diffs


def _load(path: str) -> dict:
    """Load a summary from a .json file, or extract one from a .usd*."""
    if path.endswith(".json"):
        with open(path) as fh:
            data = json.load(fh)
        return data.get("summary", data)
    return extract(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="extract a summary from a USD")
    pe.add_argument("usd")
    pe.add_argument("-o", "--output", help="write JSON here (default: stdout)")

    pc = sub.add_parser("compare", help="compare reference vs candidate (.json or .usd)")
    pc.add_argument("reference")
    pc.add_argument("candidate")
    pc.add_argument("--rtol", type=float, default=1e-3)

    args = parser.parse_args(argv)

    if args.cmd == "extract":
        summary = extract(args.usd)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            with open(args.output, "w") as fh:
                fh.write(text + "\n")
            print(f"Wrote {args.output}")
        else:
            print(text)
        return 0

    ref, cand = _load(args.reference), _load(args.candidate)
    diffs = compare(ref, cand, args.rtol)
    for label, d in (("reference", ref), ("candidate", cand)):
        print(
            f"{label}: {len(d['links'])} links, {len(d['joints'])} joints, "
            f"roots={d['articulation_roots']}"
        )
    if not diffs:
        print("\nMATCH: no physics-relevant differences within tolerance.")
        return 0
    print(f"\n{len(diffs)} difference(s):")
    for d in diffs:
        print(f"  - {d}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
