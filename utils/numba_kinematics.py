"""
Numba JIT-compiled kinematics utilities.
Runs forward/inverse kinematics fast on CPU without GPU overhead
for lightweight env resets and reward shaping.
"""
import numpy as np
import numba as nb
from numba import njit, prange
from typing import Tuple


@njit(cache=True)
def dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """
    Denavit-Hartenberg homogeneous transform matrix.
    Standard DH convention: Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ], dtype=np.float64)
    return T


@njit(cache=True)
def forward_kinematics_4dof(q: np.ndarray, dh_params: np.ndarray) -> np.ndarray:
    """
    Compute end-effector pose via FK for a 4-DOF arm.
    dh_params: (4, 4) array of [a, d, alpha, theta_offset] per joint
    q: joint angles (4,)
    Returns: 4x4 homogeneous transform (world to EE)
    """
    T = np.eye(4, dtype=np.float64)
    for i in range(4):
        a     = dh_params[i, 0]
        d     = dh_params[i, 1]
        alpha = dh_params[i, 2]
        theta_offset = dh_params[i, 3]
        T = T @ dh_transform(a, d, alpha, q[i] + theta_offset)
    return T


@njit(cache=True, parallel=True)
def batch_fk(q_batch: np.ndarray, dh_params: np.ndarray) -> np.ndarray:
    """
    Forward kinematics for a batch of joint configs.
    q_batch: (N, 4) array
    Returns: (N, 4, 4) transforms
    """
    N = q_batch.shape[0]
    results = np.zeros((N, 4, 4), dtype=np.float64)
    for i in prange(N):
        results[i] = forward_kinematics_4dof(q_batch[i], dh_params)
    return results


@njit(cache=True)
def jacobian_numerical(q: np.ndarray, dh_params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Numerical Jacobian (6 x n_dof) via finite differences.
    Returns: J where dot(J, dq) = [v_x, v_y, v_z, w_x, w_y, w_z]
    """
    n = len(q)
    J = np.zeros((6, n), dtype=np.float64)

    T0 = forward_kinematics_4dof(q, dh_params)
    p0 = T0[:3, 3]

    for i in range(n):
        q_plus = q.copy()
        q_plus[i] += eps
        T_plus = forward_kinematics_4dof(q_plus, dh_params)
        p_plus = T_plus[:3, 3]
        J[:3, i] = (p_plus - p0) / eps

    return J


@njit(cache=True)
def ik_damped_least_squares(
    target_pos: np.ndarray,
    q_init: np.ndarray,
    dh_params: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-3,
    damping: float = 0.05,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, bool]:
    """
    Damped least-squares (Levenberg-Marquardt style) IK solver.
    Fast CPU-side IK for reward shaping and env resets.
    Returns: (q_solution, converged)
    """
    q = q_init.copy()
    for _ in range(max_iter):
        T = forward_kinematics_4dof(q, dh_params)
        pos_err = target_pos - T[:3, 3]
        if np.linalg.norm(pos_err) < tol:
            return q, True

        J = jacobian_numerical(q, dh_params)
        J_pos = J[:3, :]  # position Jacobian only

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err
        A = J_pos @ J_pos.T + damping ** 2 * np.eye(3)
        dq = alpha * J_pos.T @ np.linalg.solve(A, pos_err)

        q = q + dq
        # Clip to joint limits (approximate)
        q = np.clip(q, -3.14159, 3.14159)

    return q, False


# Default 4-DOF arm DH params: [a, d, alpha, theta_offset]
DEFAULT_DH_PARAMS = np.array([
    [0.0,   0.10, np.pi/2, 0.0],
    [0.30,  0.00, 0.0,     0.0],
    [0.25,  0.00, 0.0,     0.0],
    [0.10,  0.05, 0.0,     0.0],
], dtype=np.float64)
