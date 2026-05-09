"""
physics-native-ai-trainer — Entry Point

Train robot manipulation policies in GPU-accelerated MuJoCo simulation.
Logs to W&B + MLflow. Validates with Drake + cuRobo before deployment.

Usage:
  python main.py --algo ppo --timesteps 1000000 --envs 8
  python main.py --algo sac --timesteps 500000 --device cuda
  python main.py --validate-only --checkpoint models/my_run/final_model.zip
"""

import argparse
import os
import sys
import numpy as np
import torch

from policies.sb3_trainer import train as sb3_train
from tracking.experiment_tracker import ExperimentTracker
from planning.drake_validator import DrakeMotionValidator
from distributed.ray_rollout import DistributedRolloutCollector
from utils.numba_kinematics import DEFAULT_DH_PARAMS, batch_fk


def parse_args():
    parser = argparse.ArgumentParser(description="Physics-Native AI Trainer")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac", "td3"],
                        help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=8,
                        help="Number of parallel training environments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        help="Training device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip training, only run Drake validation on checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for evaluation/validation")
    parser.add_argument("--distributed", action="store_true",
                        help="Use Ray distributed rollout workers")
    parser.add_argument("--n-ray-workers", type=int, default=16)
    parser.add_argument("--use-trl", action="store_true",
                        help="Use HuggingFace TRL transformer policy instead of SB3")
    return parser.parse_args()


def run_drake_validation(checkpoint_path: str, n_eval_episodes: int = 10):
    """Load trained policy, collect trajectories, validate with Drake."""
    print("\n[Validation] Running Drake trajectory validation...")
    from stable_baselines3 import PPO, SAC, TD3
    from envs.mujoco_env import PickAndPlaceEnv

    # Try loading as SB3 model
    for AlgoClass in [PPO, SAC, TD3]:
        try:
            model = AlgoClass.load(checkpoint_path)
            break
        except Exception:
            continue
    else:
        print(f"[Validation] Could not load checkpoint: {checkpoint_path}")
        return

    env = PickAndPlaceEnv()
    validator = DrakeMotionValidator()

    success_count = 0
    valid_count = 0

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        trajectory = [obs[:env.model.nq - 7]]  # arm qpos only
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(obs[:env.model.nq - 7])
            done = terminated or truncated

        trajectory = np.array(trajectory)
        report = validator.validate_policy_trajectory(trajectory, check_collision=False)
        if report["valid"]:
            valid_count += 1
        if info.get("success", False):
            success_count += 1

    print(f"\n[Validation] Results over {n_eval_episodes} episodes:")
    print(f"  Task success:  {success_count}/{n_eval_episodes}")
    print(f"  Drake valid:   {valid_count}/{n_eval_episodes}")


def demo_numba_kinematics():
    """Quick FK batch demo using Numba JIT."""
    print("\n[Demo] Numba batch FK on 10k configs...")
    q_batch = np.random.uniform(-np.pi, np.pi, size=(10_000, 4))
    transforms = batch_fk(q_batch, DEFAULT_DH_PARAMS)
    ee_positions = transforms[:, :3, 3]
    print(f"  Workspace X: [{ee_positions[:, 0].min():.3f}, {ee_positions[:, 0].max():.3f}]")
    print(f"  Workspace Y: [{ee_positions[:, 1].min():.3f}, {ee_positions[:, 1].max():.3f}]")
    print(f"  Workspace Z: [{ee_positions[:, 2].min():.3f}, {ee_positions[:, 2].max():.3f}]")


def main():
    args = parse_args()

    print("=" * 60)
    print("  Physics-Native AI Trainer")
    print(f"  Algo: {args.algo.upper()} | Device: {args.device}")
    print(f"  Timesteps: {args.timesteps:,} | Parallel envs: {args.envs}")
    print("=" * 60)

    # Demo Numba kinematics
    demo_numba_kinematics()

    if args.validate_only:
        if not args.checkpoint:
            print("Error: --checkpoint required with --validate-only")
            sys.exit(1)
        run_drake_validation(args.checkpoint)
        return

    if args.distributed:
        print(f"\n[Ray] Initializing {args.n_ray_workers} distributed workers...")
        collector = DistributedRolloutCollector(
            n_workers=args.n_ray_workers,
            n_steps_per_worker=256,
        )
        collector.init(seed=args.seed)
        rollouts = collector.collect_parallel()
        batch = collector.aggregate(rollouts)
        print(f"  Collected batch: obs={batch['obs'].shape}, rewards mean={batch['rewards'].mean():.4f}")
        collector.shutdown()
        return

    if args.use_trl:
        from policies.trl_policy import TRLRobotTrainer, TRLRobotConfig
        cfg = TRLRobotConfig(total_episodes=1000, device=args.device if args.device != "auto" else "cpu")
        trainer = TRLRobotTrainer(cfg)
        trainer.train()
        return

    # Standard SB3 training
    model = sb3_train(
        algo=args.algo,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        seed=args.seed,
        run_name=args.run_name,
        device=args.device,
    )

    print("\n[Main] Training complete.")

    # Validate final model
    if model is not None:
        run_drake_validation(f"models/{args.run_name or args.algo}/final_model")


if __name__ == "__main__":
    main()
