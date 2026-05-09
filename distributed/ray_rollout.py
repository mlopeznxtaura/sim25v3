"""
Ray-distributed rollout workers for parallel RL data collection.
Spawns N actors across GPUs, each running independent env instances.
Feeds experience to a centralized replay buffer.
"""
import numpy as np
import time
from typing import Optional, List, Dict, Any
import ray
from ray.util.queue import Queue


@ray.remote(num_cpus=1, num_gpus=0.1)
class RolloutWorker:
    """
    Remote Ray actor running a robot sim env and collecting rollouts.
    Communicates results back to a central collector.
    """

    def __init__(self, worker_id: int, env_config: dict, seed: int = 0):
        from envs.mujoco_env import PickAndPlaceEnv
        self.worker_id = worker_id
        self.env = PickAndPlaceEnv()
        self.env.reset(seed=seed + worker_id * 1000)
        self.total_steps = 0
        self.total_episodes = 0

    def collect_rollout(self, policy_weights: Optional[dict] = None, n_steps: int = 256) -> dict:
        """
        Collect n_steps of experience using current policy.
        Returns trajectory dict: {obs, actions, rewards, dones, infos}
        """
        obs_list, action_list, reward_list, done_list = [], [], [], []

        obs, _ = self.env.reset()
        for _ in range(n_steps):
            # Random policy (in real training: use policy_weights to reconstruct model)
            action = self.env.action_space.sample()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(done)

            obs = next_obs
            self.total_steps += 1
            if done:
                self.total_episodes += 1
                obs, _ = self.env.reset()

        return {
            "worker_id": self.worker_id,
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(action_list, dtype=np.float32),
            "rewards": np.array(reward_list, dtype=np.float32),
            "dones": np.array(done_list, dtype=bool),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
        }

    def get_stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
        }


class DistributedRolloutCollector:
    """
    Manages a pool of Ray rollout workers and aggregates experience.
    """

    def __init__(self, n_workers: int = 16, n_steps_per_worker: int = 256):
        self.n_workers = n_workers
        self.n_steps_per_worker = n_steps_per_worker
        self.workers = []
        self._initialized = False

    def init(self, env_config: dict = None, seed: int = 42):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        env_config = env_config or {}
        self.workers = [
            RolloutWorker.remote(i, env_config, seed)
            for i in range(self.n_workers)
        ]
        self._initialized = True
        print(f"[Ray] Spawned {self.n_workers} rollout workers")

    def collect_parallel(self, policy_weights: Optional[dict] = None) -> List[dict]:
        """Collect rollouts from all workers in parallel."""
        if not self._initialized:
            self.init()

        futures = [
            w.collect_rollout.remote(policy_weights, self.n_steps_per_worker)
            for w in self.workers
        ]
        rollouts = ray.get(futures)
        total_steps = sum(r["total_steps"] for r in rollouts)
        print(f"[Ray] Collected {self.n_workers * self.n_steps_per_worker} steps "
              f"({self.n_workers} workers x {self.n_steps_per_worker})")
        return rollouts

    def aggregate(self, rollouts: List[dict]) -> dict:
        """Concatenate rollouts from all workers into a single batch."""
        return {
            "obs": np.concatenate([r["obs"] for r in rollouts], axis=0),
            "actions": np.concatenate([r["actions"] for r in rollouts], axis=0),
            "rewards": np.concatenate([r["rewards"] for r in rollouts], axis=0),
            "dones": np.concatenate([r["dones"] for r in rollouts], axis=0),
        }

    def shutdown(self):
        ray.shutdown()
        print("[Ray] Workers shut down")
