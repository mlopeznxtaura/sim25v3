"""
HuggingFace TRL (PPO / DPO) policy training layer.
Wraps the robot manipulation task as a text-action interface for
training with transformer-based policies.
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch

from envs.mujoco_env import PickAndPlaceEnv


@dataclass
class TRLRobotConfig:
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    learning_rate: float = 1.41e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    max_new_tokens: int = 32
    total_episodes: int = 10_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def encode_obs_as_prompt(obs: np.ndarray) -> str:
    """Convert robot observation vector to natural language prompt for LLM policy."""
    qpos = obs[:4]
    qvel = obs[4:8]
    obj_pos = obs[8:11]
    target_pos = obs[11:14]
    return (
        f"Robot arm state: joints=[{', '.join(f'{v:.2f}' for v in qpos)}], "
        f"velocities=[{', '.join(f'{v:.2f}' for v in qvel)}]. "
        f"Object at ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f}). "
        f"Target at ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}). "
        f"What torque actions should the arm take? Output 4 floats between -1 and 1."
    )


def decode_action_from_response(text: str, n_actions: int = 4) -> np.ndarray:
    """Parse LLM text output into continuous action vector."""
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    actions = [float(x) for x in nums[:n_actions]]
    while len(actions) < n_actions:
        actions.append(0.0)
    return np.clip(np.array(actions, dtype=np.float32), -1, 1)


class TRLRobotTrainer:
    """
    Train a transformer policy on robot manipulation via PPO.
    The LLM observes robot state as text and outputs action tokens.
    """

    def __init__(self, config: TRLRobotConfig = None):
        self.cfg = config or TRLRobotConfig()

        # Load model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.cfg.model_name
        ).to(self.cfg.device)

        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.cfg.model_name
        ).to(self.cfg.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        ppo_config = PPOConfig(
            model_name=self.cfg.model_name,
            learning_rate=self.cfg.learning_rate,
            batch_size=self.cfg.batch_size,
            mini_batch_size=self.cfg.mini_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            ppo_epochs=self.cfg.ppo_epochs,
        )

        self.trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        self.env = PickAndPlaceEnv()

    def collect_trajectory(self, n_steps=16):
        """Run policy in env, collect (query, response, reward) tuples."""
        queries, responses, rewards = [], [], []

        obs, _ = self.env.reset()
        for _ in range(n_steps):
            prompt = encode_obs_as_prompt(obs)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.cfg.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response_ids = output_ids[:, input_ids.shape[1]:]
            response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
            action = decode_action_from_response(response_text)

            obs, reward, terminated, truncated, _ = self.env.step(action)

            queries.append(input_ids[0])
            responses.append(response_ids[0])
            rewards.append(torch.tensor(reward, dtype=torch.float32))

            if terminated or truncated:
                obs, _ = self.env.reset()

        return queries, responses, rewards

    def train(self):
        """Main PPO training loop."""
        print(f"[TRL] Training {self.cfg.model_name} on robot manipulation task")
        print(f"[TRL] Device: {self.cfg.device}, Episodes: {self.cfg.total_episodes}")

        for episode in range(self.cfg.total_episodes):
            queries, responses, rewards = self.collect_trajectory()
            stats = self.trainer.step(queries, responses, rewards)

            if episode % 100 == 0:
                mean_reward = torch.stack(rewards).mean().item()
                print(f"Episode {episode:5d} | mean_reward: {mean_reward:.4f} | "
                      f"ppo/loss: {stats.get('ppo/loss/total', 0):.4f}")

        print("[TRL] Training complete.")
        self.model.save_pretrained("models/trl_robot_policy")
        self.tokenizer.save_pretrained("models/trl_robot_policy")
