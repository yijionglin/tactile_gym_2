import numpy as np
from tactile_gym.rl_envs.exploration.surface_follow.base_surface_env import (
    BaseSurfaceEnv,
)

env_modes_default = {
    'movement_mode': 'xRz',
    'control_mode': 'TCP_velocity_control',
    'noise_mode': 'simplex',
    'observation_mode': 'oracle',
    'reward_mode': 'dense',
}


class SurfaceFollowVertEnv(BaseSurfaceEnv):
    def __init__(
        self,
        max_steps=200,
        image_size=[64, 64],
        env_modes=env_modes_default,
        show_gui=False,
        show_tactile=False,
    ):

        super(SurfaceFollowVertEnv, self).__init__(
            max_steps, image_size, env_modes, show_gui, show_tactile
        )

    def encode_actions(self, actions):
        """
        return actions as np.array in correct places for sending to ur5
        """
        encoded_actions = np.zeros(6)

        if self.t_s_name == "tactip":
            encoded_actions[1] = self.workframe_directions[1] * self.max_action
        if self.t_s_name == "digitac":
            encoded_actions[1] = self.workframe_directions[1] * self.max_action * 0.9
        elif self.t_s_name == "digit":
            encoded_actions[1] = self.workframe_directions[1] * self.max_action * 0.7

        if self.movement_mode == "xRz":
            encoded_actions[0] = actions[0]
            encoded_actions[5] = actions[1]

        return encoded_actions

    def sparse_reward(self):
        """
        Calculate the reward when in sparse mode.
        Reward is accumulated during an episode and given when a goal is achieved.
        """

        self.accum_rew += self.dense_reward()

        dist = self.xyz_dist_to_goal()
        if dist < self.termination_dist:
            reward = self.accum_rew
        else:
            reward = 0

        return reward

    def dense_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        # W_goal = 1.0
        W_surf = 10.0
        W_norm = 3.0

        # get the distances
        goal_dist = self.xy_dist_to_goal()
        surf_dist = self.z_dist_to_surface()
        cos_dist = self.cos_dist_to_surface_normal()

        reward = -((W_surf * surf_dist) + (W_norm * cos_dist))
        return reward

    def get_extended_feature_array(self):
        """
        features needed to help complete task.
        Goal pose and current tcp pose.
        """
        # get sim info on TCP
        (
            tcp_pos_workframe,
            _,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        # convert the features into array that matches the image observation shape
        feature_array = np.array([*tcp_pos_workframe, *self.goal_pos_workframe])

        return feature_array

    def get_act_dim(self):
        """
        Returns action dimensions, dependent on the env/task.
        """
        if self.movement_mode == "yz":
            return 2
        if self.movement_mode == "xyz":
            return 3
        if self.movement_mode == "xRz":
            return 2
        if self.movement_mode == "yzRx":
            return 3
        if self.movement_mode == "xyzRxRy":
            return 5
