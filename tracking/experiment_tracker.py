"""
Unified experiment tracking: W&B + MLflow together.
Log metrics, artifacts, model checkpoints, and environment videos.
"""
import os
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import wandb
import mlflow
import mlflow.pytorch


class ExperimentTracker:
    """
    Single interface for both W&B and MLflow tracking.
    Use for all training runs in physics-native-ai-trainer.
    """

    def __init__(
        self,
        project: str = "physics-native-ai-trainer",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = True,
        use_mlflow: bool = True,
        mlflow_uri: str = "./mlruns",
        tags: Optional[Dict[str, str]] = None,
    ):
        self.project = project
        self.run_name = run_name or f"run-{int(time.time())}"
        self.config = config or {}
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        self._step = 0

        if use_wandb:
            self.wb_run = wandb.init(
                project=project,
                name=self.run_name,
                config=config,
                tags=list(tags.values()) if tags else None,
                reinit=True,
            )
            print(f"[W&B] Run: {self.wb_run.url}")

        if use_mlflow:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(project)
            self.mlf_run = mlflow.start_run(run_name=self.run_name, tags=tags)
            if config:
                mlflow.log_params({
                    str(k): str(v)[:250]  # MLflow param value limit
                    for k, v in config.items()
                })
            print(f"[MLflow] Run ID: {self.mlf_run.info.run_id}")

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B and/or MLflow."""
        step = step if step is not None else self._step
        self._step = step + 1

        if self.use_wandb:
            wandb.log(metrics, step=step)

        if self.use_mlflow:
            for k, v in metrics.items():
                mlflow.log_metric(k.replace("/", "_"), float(v), step=step)

    def log_video(self, frames: np.ndarray, name: str = "rollout", fps: int = 30):
        """Log video of environment rollout."""
        if self.use_wandb:
            wandb.log({name: wandb.Video(frames, fps=fps, format="gif")})

    def save_model(self, model, name: str = "policy", artifact_type: str = "model"):
        """Save model checkpoint to W&B artifacts and MLflow."""
        import torch

        ckpt_path = Path(f"./checkpoints/{self.run_name}/{name}.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        if self.use_wandb:
            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(str(ckpt_path))
            self.wb_run.log_artifact(artifact)

        if self.use_mlflow:
            mlflow.pytorch.log_model(model, name)

        print(f"[Tracker] Model saved: {ckpt_path}")

    def finish(self):
        if self.use_wandb:
            self.wb_run.finish()
        if self.use_mlflow:
            mlflow.end_run()
        print("[Tracker] Run finished")
