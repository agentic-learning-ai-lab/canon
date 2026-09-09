import gymnasium as gym
from typing import Optional
import numpy as np
from utils.evaluate import EvalMetricTracker
import copy

def collect_multi_view_episode(
    env: gym.Env,
    policy_env: gym.Env,
    policy,
    metric_tracker: EvalMetricTracker,
    epsilon: float = 0.0,
    noise_type: str = "gaussian",
    init_obs: Optional[np.ndarray] = None,
    must_success: bool=False
):
    success_label = False
    time = 0
    while not success_label:
        if init_obs is None or time > 0:
            obs, _ = env.reset()
            time += 1
        else:
            obs = init_obs
            time += 1
        policy_obs = policy_env.reset()[0]

        episode_length = 0
        is_terminal = False
        trajectory = {"image": [], "state": [], "action": [], "reward": [], "is_success": True}
        metric_tracker.reset()

        while not is_terminal:
            action = policy.get_action(policy_obs)
            if noise_type == "gaussian":
                action = action + epsilon * np.random.randn(*action.shape)
            elif noise_type == "uniform":
                action = action + epsilon * policy_env.action_space.sample()
            elif noise_type == "random":
                action = policy_env.action_space.sample()
            elif noise_type == 'no_noise':
                pass
            else:
                raise ValueError("Invalid noise type provided.")

            action = np.clip(action, -1 + 1e-5, 1 - 1e-5)  # Clip the action to the valid range after noise.
            next_obs, reward, terminated, truncated, info = env.step(action)
            policy_obs, _, _, _, src_info = policy_env.step(action)
            metric_tracker.step(reward, info)
            is_terminal = terminated or truncated

            episode_length += 1

            # If the other env finishes we have to terminate
            if getattr(policy_env, "curr_path_length", 0) == policy_env.max_path_length:
                print(f"  [TERM] policy_env path length limit at step {episode_length}")
                is_terminal = True
                if 'fig_arrays' not in info:
                    info['fig_arrays'] = env.unwrapped.get_arrays()

            if episode_length == 500:
                is_terminal = True
                if 'fig_arrays' not in info:
                    info['fig_arrays'] = env.unwrapped.get_arrays()

            if next_obs.max() > 2:
                print(f"  [TERM] obs.max()={next_obs.max():.3f} > 2 at step {episode_length}")
                is_terminal = True
                if 'fig_arrays' not in info:
                    info['fig_arrays'] = env.unwrapped.get_arrays()

            trajectory['state'].append(obs)
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)
            if is_terminal:
                fig_arrays = copy.deepcopy(info['fig_arrays'])
                # Truncate images to match recorded state/action length
                num_recorded = len(trajectory['state'])
                for key in list(fig_arrays.keys()):
                    if len(fig_arrays[key]) > num_recorded:
                        fig_arrays[key] = fig_arrays[key][:num_recorded]
                if 'wrist' in fig_arrays:
                    trajectory['image_wrist'] = fig_arrays.pop('wrist')
                # Separate camera 0 (frontal view, azimuth=90) into its own entry
                if 0 in fig_arrays:
                    trajectory['image_front'] = fig_arrays.pop(0)
                trajectory['image'] = fig_arrays
                trajectory['is_success'] = bool(info['is_success'])
            obs = next_obs
        if must_success:
            success_label = trajectory['is_success']
        else:
            success_label = True

    return trajectory

def collect_multi_view_episode_v2(
    env: gym.Env,
    policy_env: gym.Env,
    policy,
    metric_tracker: EvalMetricTracker,
    epsilon: float = 0.0,
    noise_type: str = "gaussian",
    init_obs: Optional[np.ndarray] = None,
    must_success: bool=False
):
    """V2 variant: stores all camera images uniformly as image_{cam_id}.

    Unlike v1, camera 0 is not separated into a special 'image_front' key;
    it is treated identically to any other pool camera.  This matches the v2
    HDF5 schema where each trajectory has exactly image_{v1} and image_{v2}.
    """
    success_label = False
    time = 0
    while not success_label:
        if init_obs is None or time > 0:
            obs, _ = env.reset()
            time += 1
        else:
            obs = init_obs
            time += 1
        policy_obs = policy_env.reset()[0]

        episode_length = 0
        is_terminal = False
        trajectory = {"image": {}, "state": [], "action": [], "reward": [], "is_success": True,
                      "mujoco_state": []}  # qpos+qvel per step for future re-rendering
        metric_tracker.reset()

        while not is_terminal:
            action = policy.get_action(policy_obs)
            if noise_type == "gaussian":
                action = action + epsilon * np.random.randn(*action.shape)
            elif noise_type == "uniform":
                action = action + epsilon * policy_env.action_space.sample()
            elif noise_type == "random":
                action = policy_env.action_space.sample()
            elif noise_type == 'no_noise':
                pass
            else:
                raise ValueError("Invalid noise type provided.")

            action = np.clip(action, -1 + 1e-5, 1 - 1e-5)
            next_obs, reward, terminated, truncated, info = env.step(action)
            policy_obs, _, _, _, _ = policy_env.step(action)
            metric_tracker.step(reward, info)
            is_terminal = terminated or truncated

            episode_length += 1

            if getattr(policy_env, "curr_path_length", 0) == policy_env.max_path_length:
                print(f"  [TERM] policy_env path length limit at step {episode_length}")
                is_terminal = True
                if 'fig_arrays' not in info:
                    info['fig_arrays'] = env.unwrapped.get_arrays()

            if episode_length == 500:
                is_terminal = True
                if 'fig_arrays' not in info:
                    info['fig_arrays'] = env.unwrapped.get_arrays()

            if next_obs.max() > 2:
                print(f"  [TERM] obs.max()={next_obs.max():.3f} > 2 at step {episode_length}")
                is_terminal = True
                if 'fig_arrays' not in info:
                    info['fig_arrays'] = env.unwrapped.get_arrays()

            trajectory['state'].append(obs)
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)
            # Log raw MuJoCo state (qpos+qvel) at each step for future re-rendering.
            # This is distinct from the 39-dim observation vector stored in 'state'.
            trajectory['mujoco_state'].append(
                np.concatenate([env.unwrapped._env.data.qpos.copy(),
                                env.unwrapped._env.data.qvel.copy()])
            )
            if is_terminal:
                fig_arrays = copy.deepcopy(info['fig_arrays'])
                num_recorded = len(trajectory['state'])
                for key in list(fig_arrays.keys()):
                    if len(fig_arrays[key]) > num_recorded:
                        fig_arrays[key] = fig_arrays[key][:num_recorded]
                # v2: all cameras stored uniformly; wrist separated, rest kept as-is
                if 'wrist' in fig_arrays:
                    trajectory['image_wrist'] = fig_arrays.pop('wrist')
                trajectory['image'] = fig_arrays   # {cam_id: frames}; cam_0 not special
                trajectory['is_success'] = bool(info['is_success'])
            obs = next_obs
        if must_success:
            success_label = trajectory['is_success']
        else:
            success_label = True

    return trajectory


def collect_single_view_episode(
    env: gym.Env,
    policy_env: gym.Env,
    policy,
    metric_tracker: EvalMetricTracker,
    epsilon: float = 0.0,
    noise_type: str = "gaussian",
    init_obs: Optional[np.ndarray] = None,
    must_success: bool=False
):
    success_label = False
    time = 0
    while not success_label:
        if init_obs is None or time > 0:
            obs, _ = env.reset()
            time += 1
        else:
            obs = init_obs
            time += 1
        policy_obs = policy_env.reset()[0]

        episode_length = 0
        is_terminal = False
        trajectory = {"image": [], "state": [], "action": [], "reward": [], "is_success": True,
                      "mujoco_state": []}  # qpos+qvel per step for future re-rendering
        metric_tracker.reset()

        while not is_terminal:
            action = policy.get_action(policy_obs)
            if noise_type == "gaussian":
                action = action + epsilon * np.random.randn(*action.shape)
            elif noise_type == "uniform":
                action = action + epsilon * policy_env.action_space.sample()
            elif noise_type == "random":
                action = policy_env.action_space.sample()
            elif noise_type == 'no_noise':
                pass
            else:
                raise ValueError("Invalid noise type provided.")

            action = np.clip(action, -1 + 1e-5, 1 - 1e-5)  # Clip the action to the valid range after noise.
            next_obs, reward, terminated, truncated, info = env.step(action)
            policy_obs, _, _, _, src_info = policy_env.step(action)
            metric_tracker.step(reward, info)
            is_terminal = terminated or truncated

            episode_length += 1

            # If the other env finishes we have to terminate
            if getattr(policy_env, "curr_path_length", 0) == policy_env.max_path_length:
                is_terminal = True
                if 'fig_arrays' not in info:
                    info['fig_arrays'] = env.unwrapped.get_arrays()

            if episode_length == 500:
                is_terminal = True
                if 'fig_arrays' not in info:
                    info['fig_arrays'] = env.unwrapped.get_arrays()

            if next_obs.max() > 2:
                is_terminal = True
                if 'fig_arrays' not in info:
                    info['fig_arrays'] = env.unwrapped.get_arrays()

            trajectory['state'].append(obs)
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)
            # Log raw MuJoCo state (qpos+qvel) at each step for future re-rendering.
            # This is distinct from the 39-dim observation vector stored in 'state'.
            trajectory['mujoco_state'].append(
                np.concatenate([env.unwrapped._env.data.qpos.copy(),
                                env.unwrapped._env.data.qvel.copy()])
            )
            if is_terminal:
                fig_arrays = copy.deepcopy(info['fig_arrays'])
                # Truncate images to match recorded state/action length
                num_recorded = len(trajectory['state'])
                for key in list(fig_arrays.keys()):
                    if len(fig_arrays[key]) > num_recorded:
                        fig_arrays[key] = fig_arrays[key][:num_recorded]
                if 'wrist' in fig_arrays:
                    trajectory['image_wrist'] = fig_arrays.pop('wrist')
                trajectory['image'] = fig_arrays
                trajectory['is_success'] = bool(info['is_success'])
                trajectory['camera_id'] = env.unwrapped.camera_id
                trajectory['camera_config'] = env.unwrapped.camera_config
            obs = next_obs
        if must_success:
            success_label = trajectory['is_success']
        else:
            success_label = True

    return trajectory
