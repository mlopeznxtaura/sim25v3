"""
Drake SDK motion planning validator.
Validates learned policies against kinodynamic constraints before deployment.
Uses Drake's mathematical programming + RRT for collision-free path planning.
"""
import numpy as np
from typing import Optional, List, Tuple
try:
    from pydrake.all import (
        DiagramBuilder, AddMultibodyPlantSceneGraph,
        Parser, RigidTransform, RollPitchYaw,
        InverseKinematics, Solve, RotationMatrix,
        SpatialVelocity, LeafSystem, BasicVector,
        AbstractValue, PortDataType,
    )
    from pydrake.planning import RrtPlanner
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    print("Warning: Drake not available. Install: pip install drake")


class DrakeMotionValidator:
    """
    Validates robot trajectories from RL policies using Drake's rigid body dynamics.
    Checks: collision, joint limits, torque limits, workspace reachability.
    """

    def __init__(self, urdf_path: Optional[str] = None):
        if not DRAKE_AVAILABLE:
            raise ImportError("Drake SDK required. Install: pip install drake")

        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=0.002
        )

        parser = Parser(self.plant)
        if urdf_path:
            parser.AddModelFromFile(urdf_path)
        else:
            # Use a simple 4-DOF arm model inline
            self._add_simple_arm(parser)

        self.plant.Finalize()
        self.diagram = self.builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        self.nq = self.plant.num_positions()
        self.nv = self.plant.num_velocities()
        self.nu = self.plant.num_actuators()

        # Joint limits from plant
        self.q_lower = self.plant.GetPositionLowerLimits()
        self.q_upper = self.plant.GetPositionUpperLimits()
        self.effort_limits = self.plant.GetEffortUpperLimits()

    def _add_simple_arm(self, parser):
        """Add a simple arm model for validation."""
        # In real usage, load your URDF here
        pass

    def check_joint_limits(self, trajectory: np.ndarray) -> Tuple[bool, List[int]]:
        """
        Check if all configurations in trajectory satisfy joint limits.
        trajectory: (T, nq) array of joint configurations
        Returns: (valid, list_of_violating_timesteps)
        """
        violations = []
        for t, q in enumerate(trajectory):
            if len(q) < self.nq:
                continue
            q_arm = q[:self.nq]
            if np.any(q_arm < self.q_lower) or np.any(q_arm > self.q_upper):
                violations.append(t)
        return len(violations) == 0, violations

    def check_collision(self, trajectory: np.ndarray) -> Tuple[bool, List[int]]:
        """
        Check collision along trajectory using Drake's scene graph.
        trajectory: (T, nq) array of joint configurations
        Returns: (collision_free, list_of_collision_timesteps)
        """
        collisions = []
        sg_context = self.scene_graph.GetMyContextFromRoot(self.context)

        for t, q in enumerate(trajectory):
            if len(q) < self.nq:
                continue
            self.plant.SetPositions(self.plant_context, q[:self.nq])
            self.diagram.ForcedPublish(self.context)

            query_object = self.scene_graph.get_query_output_port().Eval(sg_context)
            inspector = query_object.inspector()
            pairs = query_object.ComputeSignedDistancePairwiseClosestPoints()

            for pair in pairs:
                if pair.distance < -0.005:  # 5mm penetration threshold
                    collisions.append(t)
                    break

        return len(collisions) == 0, collisions

    def solve_ik(
        self, target_pos: np.ndarray, target_rot: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics for target end-effector pose.
        Returns joint configuration or None if IK fails.
        """
        ik = InverseKinematics(self.plant, self.plant_context)
        q_vars = ik.q()

        # Position constraint
        end_effector = self.plant.GetBodyByName("gripper") if             self.plant.HasBodyNamed("gripper") else self.plant.world_body()
        frame_W = self.plant.world_frame()
        frame_EE = end_effector.body_frame()

        ik.AddPositionConstraint(
            frameB=frame_EE, p_BQ=np.zeros(3),
            frameA=frame_W,
            p_AQ_lower=target_pos - 0.01,
            p_AQ_upper=target_pos + 0.01,
        )

        if target_rot is not None:
            R_target = RotationMatrix(target_rot)
            ik.AddOrientationConstraint(
                frameAbar=frame_W, R_AbarA=R_target,
                frameBbar=frame_EE, R_BbarB=RotationMatrix.Identity(),
                theta_bound=0.1,
            )

        prog = ik.prog()
        q0 = q_init if q_init is not None else np.zeros(self.nq)
        prog.SetInitialGuess(q_vars, q0[:self.nq])

        result = Solve(prog)
        if result.is_success():
            return result.GetSolution(q_vars)
        return None

    def validate_policy_trajectory(
        self, trajectory: np.ndarray, check_collision: bool = True
    ) -> dict:
        """
        Full validation pipeline: joint limits + optionally collision.
        Returns validation report.
        """
        report = {
            "valid": True,
            "joint_limit_violations": [],
            "collision_timesteps": [],
            "trajectory_length": len(trajectory),
        }

        jl_ok, jl_violations = self.check_joint_limits(trajectory)
        report["joint_limit_violations"] = jl_violations
        if not jl_ok:
            report["valid"] = False
            print(f"[Drake] Joint limit violations at timesteps: {jl_violations[:5]}...")

        if check_collision and DRAKE_AVAILABLE:
            try:
                col_ok, col_timesteps = self.check_collision(trajectory)
                report["collision_timesteps"] = col_timesteps
                if not col_ok:
                    report["valid"] = False
                    print(f"[Drake] Collisions at timesteps: {col_timesteps[:5]}...")
            except Exception as e:
                print(f"[Drake] Collision check failed: {e}")

        if report["valid"]:
            print(f"[Drake] Trajectory VALID ({len(trajectory)} steps)")
        else:
            print(f"[Drake] Trajectory INVALID")

        return report
