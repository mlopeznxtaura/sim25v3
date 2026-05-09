# Physics-Native AI Trainer

Cluster 02 of the NextAura 500 SDKs / 25 Clusters project.

Train robot manipulation policies directly inside GPU-accelerated physics simulation — no real hardware needed.

## Architecture

- MuJoCo / Isaac Lab physics backend
- Gaussian Splatting for real-environment visual context
- Stable-Baselines3 + HuggingFace TRL for policy training
- Drake + cuRobo for motion planning validation
- Weights & Biases for experiment tracking
- MLflow for model registry
- Ray for distributed rollouts

## SDKs Used

MuJoCo SDK, NVIDIA Isaac Lab, Warp (NVIDIA), Gymnasium, Stable-Baselines3, PyTorch, cuDNN, Weights & Biases, Pinocchio SDK, cuRobo, Drake SDK, HuggingFace TRL, RAPIDS RAFT, Ray SDK, MLflow, Numba SDK, SymPy, Open3D SDK, Gaussian Splatting, nerfstudio

## Quickstart

```bash
pip install -r requirements.txt
python main.py --env PickAndPlace-v1 --algo ppo --timesteps 1000000
```

## Structure

```
envs/          MuJoCo + Gymnasium wrapped environments
policies/      SB3 + TRL policy training
planning/      Drake + cuRobo motion validation
perception/    Gaussian Splatting + Open3D scene reconstruction
distributed/   Ray rollout workers
tracking/      W&B + MLflow experiment logging
utils/         Numba JIT helpers, SymPy kinematics
main.py        Entry point
```
