"""
cuRobo GPU-accelerated motion planning.
Parallelized trajectory optimization on GPU — generates collision-free,
torque-limited trajectories at real-time speeds.
"""
import numpy as np
import torch
from typing import Optional, List, Dict, Any
try:
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.types.math import Pose
    from curobo.geom.sdf.world import CollisionCheckerType
    CUROBO_AVAILABLE = True
except ImportError:
    CUROBO_AVAILABLE = False
    print("Warning: cuRobo not available. Install via: pip install curobo-torch")


class CuRoboPlanner:
    """
    GPU-accelerated motion planner using cuRobo.
    Generates collision-free trajectories 100x faster than CPU planners.
    Designed for real-time replanning during RL policy rollouts.
    """

    def __init__(
        self,
        robot_cfg_path: str = "configs/robot.yml",
        world_cfg_path: str = "configs/world.yml",
        device: str = "cuda",
        num_trajopt_seeds: int = 12,
        num_graph_seeds: int = 12,
    ):
        if not CUROBO_AVAILABLE:
            raise ImportError("cuRobo required. See: https://curobo.org/get_started/1_install_instructions.html")

        self.device = device
        self.tensor_args = TensorDeviceType(device=device)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg_path,
            world_cfg_path,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=num_trajopt_seeds,
            num_graph_seeds=num_graph_seeds,
            interpolation_dt=0.02,
            collision_cache={"obb": 10, "mesh": 10},
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        print(f"[cuRobo] Motion planner ready on {device}")

    def plan_to_pose(
        self,
        start_q: np.ndarray,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
        max_attempts: int = 3,
    ) -> Optional[np.ndarray]:
        """
        Plan collision-free trajectory from start joint config to target EE pose.
        Returns (T, nq) trajectory or None if planning fails.
        """
        if target_quat is None:
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity

        goal_pose = Pose(
            position=self.tensor_args.to_device(torch.tensor(target_pos, dtype=torch.float32)),
            quaternion=self.tensor_args.to_device(torch.tensor(target_quat, dtype=torch.float32)),
        )

        start_state = self.tensor_args.to_device(
            torch.tensor(start_q, dtype=torch.float32).unsqueeze(0)
        )

        plan_config = MotionGenPlanConfig(
            max_attempts=max_attempts,
            enable_graph=True,
            enable_opt=True,
        )

        result = self.motion_gen.plan_single(start_state, goal_pose, plan_config)

        if result.success.item():
            traj = result.get_interpolated_plan()
            return traj.position.cpu().numpy()  # (T, nq)
        else:
            print(f"[cuRobo] Planning failed. Status: {result.status}")
            return None

    def batch_plan(
        self,
        start_configs: np.ndarray,
        target_positions: np.ndarray,
    ) -> List[Optional[np.ndarray]]:
        """
        Plan for a batch of (start, goal) pairs simultaneously on GPU.
        Enables massive parallelism for RL data collection.
        """
        B = len(start_configs)
        results = []

        start_tensor = self.tensor_args.to_device(
            torch.tensor(start_configs, dtype=torch.float32)
        )
        goal_poses = Pose(
            position=self.tensor_args.to_device(
                torch.tensor(target_positions, dtype=torch.float32)
            ),
            quaternion=self.tensor_args.to_device(
                torch.ones((B, 4), dtype=torch.float32) * torch.tensor([1, 0, 0, 0])
            ),
        )

        batch_result = self.motion_gen.plan_batch(
            start_tensor, goal_poses,
            MotionGenPlanConfig(max_attempts=3, enable_graph=True),
        )

        for i in range(B):
            if batch_result.success[i].item():
                traj = batch_result.get_interpolated_plan()
                results.append(traj.position[i].cpu().numpy())
            else:
                results.append(None)

        print(f"[cuRobo] Batch planning: {sum(r is not None for r in results)}/{B} succeeded")
        return results
