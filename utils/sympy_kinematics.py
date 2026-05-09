"""
SymPy symbolic kinematics for robot arm.
Derive exact Jacobians, workspace bounds, and singularity conditions analytically.
Used for design-time analysis, not real-time control.
"""
from sympy import (
    symbols, cos, sin, pi, Matrix, eye, zeros,
    simplify, trigsimp, lambdify, pprint, latex
)
from sympy.physics.mechanics import dynamicsymbols
from typing import Callable, Tuple
import numpy as np


def dh_matrix_symbolic(a, d, alpha, theta):
    """Symbolic DH homogeneous transform."""
    return Matrix([
        [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
        [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0,           sin(alpha),             cos(alpha),            d           ],
        [0,           0,                      0,                     1           ],
    ])


def build_4dof_arm_symbolic():
    """
    Build full symbolic kinematic model of 4-DOF arm.
    Returns: (q_syms, T_ee, J_sym, T_lambdified)
    """
    q1, q2, q3, q4 = symbols('q1 q2 q3 q4', real=True)
    q = Matrix([q1, q2, q3, q4])

    # DH parameters [a, d, alpha, theta_offset + qi]
    dh = [
        (0,    0.10, pi/2,  q1),
        (0.30, 0.00, 0,     q2),
        (0.25, 0.00, 0,     q3),
        (0.10, 0.05, 0,     q4),
    ]

    T = eye(4)
    T_list = []
    for a, d, alpha, theta in dh:
        Ti = dh_matrix_symbolic(a, d, alpha, theta)
        T = T * Ti
        T_list.append(T)

    T_ee = T  # End-effector transform in world frame

    # Geometric Jacobian
    z_axes = [eye(4)[:3, 2]]
    p_ee = T_ee[:3, 3]
    origins = [Matrix([0, 0, 0])]
    for Ti in T_list[:-1]:
        z_axes.append(Ti[:3, 2])
        origins.append(Ti[:3, 3])

    J = zeros(6, 4)
    for i in range(4):
        # Revolute joint Jacobian column
        z = z_axes[i]
        p = origins[i]
        Jv = z.cross(p_ee - p)  # Linear velocity
        Jw = z                   # Angular velocity
        J[:3, i] = Jv
        J[3:, i] = Jw

    return q, T_ee, J, T_list


def get_lambdified_fk(T_ee, q_syms):
    """Convert symbolic FK to fast numpy function."""
    q1, q2, q3, q4 = q_syms
    return lambdify([q1, q2, q3, q4], T_ee, modules="numpy")


def get_lambdified_jacobian(J, q_syms):
    """Convert symbolic Jacobian to fast numpy function."""
    q1, q2, q3, q4 = q_syms
    return lambdify([q1, q2, q3, q4], J, modules="numpy")


def analyze_singularities(J, q_syms):
    """
    Find singular configurations (det(J_pos * J_pos^T) = 0).
    Prints conditions analytically.
    """
    from sympy import det, solve
    J_pos = J[:3, :]
    manipulability = det(J_pos * J_pos.T)
    manip_simplified = trigsimp(manipulability)
    print("Manipulability measure (det(J J^T)):")
    pprint(manip_simplified)
    return manip_simplified


def workspace_monte_carlo(fk_fn: Callable, n_samples: int = 100_000) -> np.ndarray:
    """
    Estimate robot workspace by random joint sampling.
    Returns (N, 3) array of reachable end-effector positions.
    """
    q_samples = np.random.uniform(-np.pi, np.pi, size=(n_samples, 4))
    positions = np.zeros((n_samples, 3))
    for i, q in enumerate(q_samples):
        T = np.array(fk_fn(*q), dtype=np.float64)
        positions[i] = T[:3, 3]
    return positions
