from __future__ import annotations

import numpy as np

from urdf2any.transforms import (
    Transform,
    diagonalize_inertia,
    quat_align_x_to,
    quat_from_rpy,
    quat_multiply,
    quat_rotate,
)


def test_rpy_zero_is_identity():
    assert np.allclose(quat_from_rpy(0, 0, 0), [1, 0, 0, 0])


def test_rpy_yaw_90_rotates_x_to_y():
    q = quat_from_rpy(0, 0, np.pi / 2)
    assert np.allclose(quat_rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-9)


def test_transform_compose_and_inverse():
    t = Transform(translation=np.array([1.0, 2.0, 3.0]), rotation=quat_from_rpy(0, 0, np.pi / 2))
    identity = t @ t.inverse()
    assert np.allclose(identity.translation, 0, atol=1e-9)
    assert np.allclose(quat_rotate(identity.rotation, [1, 0, 0]), [1, 0, 0], atol=1e-9)


def test_compose_chains_translation_through_rotation():
    rot = Transform(rotation=quat_from_rpy(0, 0, np.pi / 2))
    trans = Transform(translation=np.array([1.0, 0.0, 0.0]))
    composed = rot @ trans
    assert np.allclose(composed.translation, [0, 1, 0], atol=1e-9)


def test_align_x_to_axis():
    for axis in ([0, 1, 0], [0, 0, 1], [1, 1, 0], [-1, 0, 0]):
        q = quat_align_x_to(np.array(axis, dtype=float))
        expected = np.array(axis, dtype=float)
        expected /= np.linalg.norm(expected)
        assert np.allclose(quat_rotate(q, [1, 0, 0]), expected, atol=1e-9)


def test_diagonalize_already_diagonal():
    inertia = np.diag([0.01, 0.02, 0.03])
    diag, quat = diagonalize_inertia(inertia)
    assert np.allclose(sorted(diag), [0.01, 0.02, 0.03])
    assert np.allclose(quat_multiply(quat, [1, 0, 0, 0])[0], quat[0])


def test_diagonalize_rotated_inertia_recovers_eigenvalues():
    base = np.diag([0.01, 0.02, 0.03])
    r = quat_from_rpy(0.3, 0.5, 0.7)
    rot = np.column_stack([quat_rotate(r, e) for e in np.eye(3)])
    rotated = rot @ base @ rot.T
    diag, _ = diagonalize_inertia(rotated)
    assert np.allclose(sorted(diag), [0.01, 0.02, 0.03], atol=1e-9)
