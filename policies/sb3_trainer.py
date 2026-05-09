"""
Stable-Baselines3 PPO/SAC/TD3 training on MuJoCo envs.
Logs everything to W&B and MLflow simultaneously.
"""
import os
import argparse
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import mlflow
import mlflow.pytorch
import torch

from envs.mujoco_env import PickAndPlaceEnv


ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}

HYPERPARAMS = {
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy": "MlpPolicy",
    },
    "sac": {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy": "MlpPolicy",
    },
    "td3": {
        "learning_rate": 1e-3,
        "buffer_size": 1_000_000,
        "batch_size": 100,
        "tau": 0.005,
        "gamma": 0.99,
        "policy": "MlpPolicy",
    },
}


class MLflowCallback(BaseCallback):
    """Log SB3 metrics to MLflow during training."""

    def __init__(self, run, verbose=0):
        super().__init__(verbose)
        self.run = run

    def _on_step(self):
        if self.n_calls % 1000 == 0:
            for key, val in self.logger.name_to_value.items():
                mlflow.log_metric(key.replace("/", "_"), val, step=self.num_timesteps)
        return True


def train(
    algo="ppo",
    total_timesteps=1_000_000,
    n_envs=8,
    seed=42,
    run_name=None,
    checkpoint_freq=50_000,
    eval_freq=10_000,
    device="auto",
):
    run_name = run_name or f"{algo}-pick-place-{seed}"

    # W&B init
    wb_run = wandb.init(
        project="physics-native-ai-trainer",
        name=run_name,
        config={
            "algo": algo,
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "seed": seed,
            **HYPERPARAMS[algo],
        },
        sync_tensorboard=True,
    )

    # MLflow init
    mlflow.set_experiment("physics-native-ai-trainer")
    with mlflow.start_run(run_name=run_name) as mlf_run:
        mlflow.log_params({"algo": algo, "total_timesteps": total_timesteps, **HYPERPARAMS[algo]})

        # Vectorized training envs
        vec_env = make_vec_env(
            PickAndPlaceEnv,
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
            seed=seed,
        )

        # Eval env (single)
        eval_env = Monitor(PickAndPlaceEnv())

        AlgoClass = ALGOS[algo]
        hparams = {k: v for k, v in HYPERPARAMS[algo].items() if k != "policy"}
        model = AlgoClass(
            HYPERPARAMS[algo]["policy"],
            vec_env,
            verbose=1,
            seed=seed,
            device=device,
            tensorboard_log=f"./runs/{run_name}",
            **hparams,
        )

        callbacks = [
            WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f"models/{run_name}",
                verbose=2,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=f"models/{run_name}/best",
                log_path=f"logs/{run_name}",
                eval_freq=eval_freq,
                n_eval_episodes=10,
                deterministic=True,
                render=False,
            ),
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=f"checkpoints/{run_name}",
                name_prefix=algo,
            ),
            MLflowCallback(run=mlf_run),
        ]

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        # Save final model
        model_path = f"models/{run_name}/final_model"
        model.save(model_path)
        mlflow.pytorch.log_model(model.policy, "policy")
        print(f"[trainer] Model saved to {model_path}")

    wb_run.finish()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="ppo", choices=ALGOS.keys())
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = train(
        algo=args.algo,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        seed=args.seed,
        device=args.device,
    )
