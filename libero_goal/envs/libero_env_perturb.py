import os
import numpy as np
from . import benchmark
from . import get_libero_path
from .libero_env import LiberoEnv, IMAGE_SIZE
from .envs.env_wrapper import OffScreenRenderEnv


class LiberoEnvPerturb(LiberoEnv):

    def __init__(
        self, task_suite_name="libero_goal", image_size=IMAGE_SIZE, id="libero_goal",
        view_indices=None, view_interval=None, mixed_view=False, single_view=None, camera_distance=None
        # rotation=None
    ):
        super().__init__(task_suite_name, image_size, id)
        
        self.mixed_view = mixed_view
        self.view_idx = -1
        self.view_interval = view_interval
        self.camera_distance = camera_distance

        if mixed_view:
            view_indices = list(view_indices)
            if view_interval is not None:
                assert isinstance(view_indices, list) and len(view_indices) == 2, f'view_indices: expected List[2]; received {view_indices} ({type(view_indices)})'
                self.make_views(view_indices)
            else:
                self.views = view_indices
            print(f'LiberoEnvPerturb: view_indices={view_indices} view_interval={view_interval}')
        else:
            assert isinstance(view_indices, (int, float)), f'view_indices = {view_indices}: expected number (int or float): got {type(view_indices)} {view_indices}'
            self.views = [view_indices]
        self.single_view = single_view

    def get_angles(self, view_range):
        """
        Generate a list of angles between start and end at every interval step.
        Handles wrapping around 360°/0° and negative degrees.
        
        Args:
            start (float): Starting angle in degrees
            end (float): Ending angle in degrees
            interval (float): Step size in degrees
        
        Returns:
            list: List of angles in the range [start, end] at interval steps
        """
        start, end = view_range
        interval = self.view_interval

        # Normalize angles to [0, 360)
        start_norm = start % 360
        end_norm = end % 360
        
        # Calculate the total angular distance (accounting for wrap-around)
        if end_norm >= start_norm:
            angular_distance = end_norm - start_norm
        else:
            angular_distance = (360 - start_norm) + end_norm
        
        # Generate angles
        num_steps = int(angular_distance / interval) + 1
        angles = [(start_norm + i * interval) % 360 for i in range(num_steps)]
        
        return angles

    def make_views(self, view_range):

        self.views = self.get_angles(view_range)

    def incr_view(self):
        self.view_idx = (self.view_idx + 1) % len(self.views)

    def step(self, action):

        obs, reward, done, info = super().step(action)
        
        if self.single_view:
            obs = obs[:1] # only obs["agentview_image"] # libero_env.py L127
            info['image'] = info['image'][:, :IMAGE_SIZE] # also crop robot_arm_view from viz

        return obs, reward, done, info

    def reset(self, goal_idx, seed=None):

        self.episodes += 1
        self.goal_idx = goal_idx
        self.steps = 0
        task_name = self.task_names[goal_idx]
        task_bddl_file = self._get_task_bddl_file(task_name)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.image_size,
            "camera_widths": self.image_size,
            "view_angle": self.views[self.view_idx],
            "camera_distance": self.camera_distance
        }

        self.env = OffScreenRenderEnv(**env_args)

        self.env.seed(self._seed + self.episodes)
        obs = self.env.reset()
        zero_action = np.zeros(7)
        for i in range(20):
            obs, _, _, _ = self.env.step(zero_action)  # make sure objects are stable
        self.last_obs = obs # Store last raw observation
        self.finished_tasks = {task_name: False for task_name in self.task_names}

        obs = self._get_img_obs(obs)
        if self.single_view:
            obs = obs[:1] # only obs["agentview_image"] # libero_env.py L127

        return (obs / 255.0).astype(np.float32)
