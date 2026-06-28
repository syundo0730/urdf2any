"""Rigid transform and rotation utilities.

Conventions:
- Quaternions are numpy arrays in (w, x, y, z) order.
- All quantities are SI (meters, radians) unless noted otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Quaternion for URDF fixed-axis roll-pitch-yaw (extrinsic X-Y-Z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, *v])
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[1:]


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2
    q = np.empty(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def quat_align_x_to(axis: np.ndarray) -> np.ndarray:
    """Quaternion rotating the +X unit vector onto ``axis`` (normalized)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    x = np.array([1.0, 0.0, 0.0])
    d = float(np.dot(x, a))
    if d > 1.0 - 1e-9:
        return quat_identity()
    if d < -1.0 + 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about Z
    n = np.cross(x, a)
    n = n / np.linalg.norm(n)
    half = np.arccos(np.clip(d, -1.0, 1.0)) / 2
    return np.array([np.cos(half), *(np.sin(half) * n)])


@dataclass
class Transform:
    """Rigid transform: rotation followed by translation."""

    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=quat_identity)  # (w, x, y, z)

    def __matmul__(self, other: Transform) -> Transform:
        return Transform(
            translation=self.translation + quat_rotate(self.rotation, other.translation),
            rotation=quat_multiply(self.rotation, other.rotation),
        )

    def inverse(self) -> Transform:
        inv_rot = quat_conjugate(self.rotation)
        return Transform(translation=-quat_rotate(inv_rot, self.translation), rotation=inv_rot)

    @staticmethod
    def identity() -> Transform:
        return Transform()


def diagonalize_inertia(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a symmetric 3x3 inertia tensor.

    Returns (principal moments, principal-axes quaternion). The quaternion maps
    the principal frame into the input frame.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    # eigh returns an orthogonal matrix that may be a reflection; force a rotation
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] = -eigenvectors[:, 0]
    return eigenvalues, quat_from_matrix(eigenvectors)
