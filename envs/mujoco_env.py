"""
MuJoCo pick-and-place environment wrapped as a Gymnasium env.
Uses MuJoCo 3.x + Gymnasium 0.29 API.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


class PickAndPlaceEnv(gym.Env):
    """
    Robotic arm pick-and-place task using MuJoCo physics.
    Observation: [arm_qpos(7), arm_qvel(7), object_pos(3), target_pos(3)] = 20-dim
    Action: joint torques (7-dim), continuous [-1, 1]
    Reward: shaped = -dist(gripper, object) - dist(object, target) + success_bonus
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, xml_path=None):
        self.render_mode = render_mode
        self._model_xml = xml_path or self._default_xml()
        self.model = mujoco.MjModel.from_xml_string(self._model_xml)
        self.data = mujoco.MjData(self.model)

        n_joints = self.model.nq
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_joints * 2 + 6,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )
        self._viewer = None
        self._step_count = 0
        self._max_steps = 500

        # Target position (randomized on reset)
        self.target_pos = np.array([0.5, 0.2, 0.1])

    def _default_xml(self):
        return """
        <mujoco model="pick_place">
          <option timestep="0.002" gravity="0 0 -9.81"/>
          <worldbody>
            <light pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="2 2 0.1" rgba=".8 .8 .8 1"/>
            <body name="arm_base" pos="0 0 0.1">
              <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <geom type="cylinder" size="0.05 0.1" rgba=".3 .3 .8 1"/>
              <body name="arm_link1" pos="0 0 0.2">
                <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                <geom type="capsule" size="0.04 0.15" rgba=".3 .5 .8 1"/>
                <body name="arm_link2" pos="0 0 0.3">
                  <joint name="joint3" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                  <geom type="capsule" size="0.035 0.12" rgba=".3 .6 .7 1"/>
                  <body name="gripper" pos="0 0 0.25">
                    <joint name="joint4" type="slide" axis="0 0 1" range="-0.05 0.05"/>
                    <geom type="box" size="0.04 0.04 0.04" rgba=".8 .4 .2 1"/>
                  </body>
                </body>
              </body>
            </body>
            <body name="object" pos="0.3 0.0 0.05">
              <freejoint/>
              <geom type="box" size="0.03 0.03 0.03" rgba=".9 .2 .2 1" mass="0.1"/>
            </body>
          </worldbody>
          <actuator>
            <motor joint="joint1" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
            <motor joint="joint2" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
            <motor joint="joint3" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
            <motor joint="joint4" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>
        """

    def _get_obs(self):
        obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        obj_pos = self.data.xpos[obj_body_id].copy()
        obs = np.concatenate([
            self.data.qpos[:self.model.nq - 7].copy(),   # arm joints
            self.data.qvel[:self.model.nv - 6].copy(),   # arm velocities
            obj_pos,
            self.target_pos
        ]).astype(np.float32)
        return obs

    def _compute_reward(self, obs, info):
        obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        obj_pos = self.data.xpos[obj_body_id].copy()

        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        gripper_pos = self.data.xpos[gripper_id].copy()

        dist_to_obj = np.linalg.norm(gripper_pos - obj_pos)
        dist_to_target = np.linalg.norm(obj_pos - self.target_pos)

        reward = -dist_to_obj * 0.5 - dist_to_target
        success = dist_to_target < 0.05
        if success:
            reward += 10.0
        info["success"] = success
        info["dist_to_target"] = float(dist_to_target)
        return float(reward), success

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize target
        rng = np.random.default_rng(seed)
        self.target_pos = rng.uniform([0.2, -0.3, 0.05], [0.6, 0.3, 0.05])

        # Randomize object start
        obj_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint") if             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint") >= 0 else -1

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        info = {}
        reward, success = self._compute_reward(obs, info)
        terminated = success
        truncated = self._step_count >= self._max_steps

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def make_env(env_id="PickAndPlace-v1", rank=0, seed=0):
    """Factory for vectorized envs with Ray / SB3."""
    def _init():
        env = PickAndPlaceEnv()
        env.reset(seed=seed + rank)
        return env
    return _init
