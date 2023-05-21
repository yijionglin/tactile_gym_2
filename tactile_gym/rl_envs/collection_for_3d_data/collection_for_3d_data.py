import gym
import numpy as np
import os
from tactile_gym.robots.arms.robot import Robot
from tactile_gym.assets import add_assets_path
from tactile_gym.rl_envs.base_tactile_env import BaseTactileEnv
from tactile_gym.rl_envs.collection_for_3d_data.rest_poses import rest_poses_dict
from ipdb import set_trace
import cv2
from tactile_gym.mesh_utils import utils_sample, utils_mesh, utils_raycasting
import trimesh

env_modes_default = {
    'movement_mode': 'xy',
    'control_mode': 'TCP_velocity_control',
    'noise_mode': 'fixed_height',
    'observation_mode': 'oracle',
    'reward_mode': 'dense',
    'arm_type': 'mg400',
}


class CollectionFor3DData(BaseTactileEnv):
    def __init__(
        self,
        max_steps=250,
        image_size=[64, 64],
        env_modes=env_modes_default,
        show_gui=False,
        show_tactile=False,
        total_collected_number = 2,
        user_mode = 'auto',
    ):

        # variables for 3d data collection

        
        self.collected_3d_data_poses_workframe = []
        self.collected_3d_data_poses_worldframe = []
        self.collected_number = 0
        self.total_collected_number = total_collected_number
        self.data_path = r'C:\Users\yijio\dev\tactip\tac_gym_3d_re\tactile_gym\examples\sim_data_for_3d'
        self.images_path = os.path.join(self.data_path, 'images')
        self.obj_scale = 0.2
        self.if_fault_collected_img = False
        # Set number of points to consider the touch chart collection valid.
        self.num_valid_points = 250
        # self.user_mode == 'manual'
        self.user_mode = user_mode
        # self.poses_path = os.path.join(self.data_path, 'poses.npy')

        # used to setup control of robot
        self._sim_time_step = 1.0 / 240.0
        self._control_rate = 1.0 / 10.0
        self._velocity_action_repeat = int(
            np.floor(self._control_rate / self._sim_time_step)
        )
        self._max_blocking_pos_move_steps = 10

        super(CollectionFor3DData, self).__init__(
            max_steps, image_size, show_gui, show_tactile, arm_type=env_modes['arm_type']
        )

        # set modes for easy adjustment
        self.movement_mode = env_modes["movement_mode"]
        self.control_mode = env_modes["control_mode"]
        self.noise_mode = env_modes["noise_mode"]
        self.observation_mode = env_modes["observation_mode"]
        self.reward_mode = env_modes["reward_mode"]
        
        # set which robot arm to use
        self.arm_type = env_modes["arm_type"]
        # self.arm_type = "ur5"
        # self.arm_type = "mg400"
        # self.arm_type = 'franka_panda'
        # self.arm_type = 'kuka_iiwa'

        # which t_s to use
        self.t_s_name = env_modes["tactile_sensor_name"]
        # self.t_s_name = 'tactip'
        # self.t_s_name = 'digit'
        self.t_s_type = "standard"
        # self.t_s_type = "mini_standard"
        self.t_s_core = "no_core"

        # distance from goal to cause termination
        self.termination_dist = 0.01

        # limits
        TCP_lims = np.zeros(shape=(6, 2))

        # TCP_lims[0, 0], TCP_lims[0, 1] = -0.175, +0.175  # x lims
        # TCP_lims[1, 0], TCP_lims[1, 1] = -0.175, +0.175  # y lims
        # TCP_lims[2, 0], TCP_lims[2, 1] = -0.1, +0.1  # z lims
        # TCP_lims[3, 0], TCP_lims[3, 1] = -np.pi / 4, +np.pi / 4  # roll lims
        # TCP_lims[4, 0], TCP_lims[4, 1] = -np.pi / 4, +np.pi / 4  # pitch lims
        # TCP_lims[5, 0], TCP_lims[5, 1] = -np.pi, np.pi  # yaw lims
        TCP_lims[0, 0], TCP_lims[0, 1] = -np.inf, +np.inf  # x lims
        TCP_lims[1, 0], TCP_lims[1, 1] = -np.inf, +np.inf  # y lims
        TCP_lims[2, 0], TCP_lims[2, 1] = -np.inf, +np.inf  # z lims
        TCP_lims[3, 0], TCP_lims[3, 1] = -np.inf, +np.inf  # roll lims
        TCP_lims[4, 0], TCP_lims[4, 1] = -np.inf, +np.inf  # pitch lims
        TCP_lims[5, 0], TCP_lims[5, 1] = -np.inf, +np.inf  # yaw lims
        # how much penetration of the tip to optimize for
        # randomly vary this on each episode
        if self.t_s_name == 'tactip':
            self.embed_dist = 0.0035
        elif self.t_s_name == 'digit':
            self.embed_dist = 0.0035
        elif self.t_s_name == 'digitac':
            self.embed_dist = 0.0035

        # setup variables
        self.setup_obj()
        self.setup_action_space()

        # load environment objects
        self.load_environment()
        self.load_obj()

        # work frame origin
        # self.workframe_pos = np.array([0.65, 0.0, 0.205])
        self.workframe_pos = np.array([0.65, 0.0, 0.35])
        self.workframe_rpy = np.array([-np.pi, 0.0, np.pi / 2])

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[self.arm_type][self.t_s_name][self.t_s_type]

        # load the ur5 with a t_s attached
        self.robot = Robot(
            self._pb,
            rest_poses=rest_poses,
            workframe_pos=self.workframe_pos,
            workframe_rpy=self.workframe_rpy,
            TCP_lims=TCP_lims,
            image_size=image_size,
            turn_off_border=True,
            arm_type=self.arm_type,
            t_s_name=self.t_s_name,
            t_s_type=self.t_s_type,
            t_s_core=self.t_s_core,
            t_s_dynamics={'stiffness': 50, 'damping': 100, 'friction': 10.0},
            show_gui=self._show_gui,
            show_tactile=self._show_tactile,
            user_mode=user_mode,
        )

        self.robot.stop_at_touch =True

        # this is needed to set some variables used for initial observation/obs_dim()
        self.reset()

        # set the observation space dependent on
        self.setup_observation_space()

    def setup_action_space(self):

        # these are used for bounds on the action space in SAC and clipping
        # range for PPO
        self.min_action, self.max_action = -0.25, 0.25

        # define action ranges per act dim to rescale output of policy
        if self.control_mode == "TCP_position_control":

            max_pos_change = 0.001  # m per step
            max_ang_change = 1 * (np.pi / 180)  # rad per step

            self.x_act_min, self.x_act_max = -max_pos_change, max_pos_change
            self.y_act_min, self.y_act_max = -max_pos_change, max_pos_change
            self.z_act_min, self.z_act_max = -max_pos_change, max_pos_change
            self.roll_act_min, self.roll_act_max = -max_ang_change, max_ang_change
            self.pitch_act_min, self.pitch_act_max =  -max_ang_change, max_ang_change
            self.yaw_act_min, self.yaw_act_max = -max_ang_change, max_ang_change

        elif self.control_mode == "TCP_velocity_control":

            max_pos_vel = 0.01  # m/s
            max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = -max_pos_vel, max_pos_vel
            self.roll_act_min, self.roll_act_max = -max_ang_vel, max_ang_vel
            self.pitch_act_min, self.pitch_act_max =  -max_ang_vel, max_ang_vel
            self.yaw_act_min, self.yaw_act_max = -max_ang_vel, max_ang_vel

        # setup action space
        self.act_dim = self.get_act_dim()
        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.act_dim,),
            dtype=np.float32,
        )

    def setup_rgb_obs_camera_params(self):
        """
        set the RGB camera position to capture full image of the env task.
        """
        # front view
        if self.arm_type == "mg400":

            self.rgb_cam_pos = [-0.20, -0.0, -0.25]
            self.rgb_cam_dist = 0.85
            self.rgb_cam_roll = 0
        else:
            self.rgb_cam_pos = [0.35, 0.0, -0.25]
            self.rgb_cam_dist = 0.75
        self.rgb_cam_yaw = 90
        self.rgb_cam_pitch = -35
        self.rgb_image_size = self._image_size
        # self.rgb_image_size = [512,512]
        self.rgb_fov = 75
        self.rgb_near_val = 0.1
        self.rgb_far_val = 100


    def setup_obj(self):
        # define an initial position for the objects (world coords)
        # self.obj_pos = [0.65, 0.0, 0.08]
        self.obj_pos = [0.5, 0.0, 0.00]
        self.obj_rpy = [np.pi/2, 0, -np.pi/2]
        self.obj_orn = self._pb.getQuaternionFromEuler(self.obj_rpy)

    def load_obj(self):
        # load temp obj and goal indicators so they can be more conveniently updated

        self.obj_path = 'c:\\users\\yijio\\dev\\tactip\\tac_gym_3d_re\\deepsdf\\data\\ShapeNetCoreV2urdf\\real_mesh\\bottle'
        self.obj_urdf_path = os.path.join(self.obj_path,'model.urdf')
        self.obj_id = self._pb.loadURDF(
            add_assets_path(self.obj_urdf_path),
            self.obj_pos,
            self.obj_orn,
            useFixedBase=True,
            flags=self._pb.URDF_INITIALIZE_SAT_FEATURES,
            globalScaling=self.obj_scale
        )
        self.goal_indicator = self._pb.loadURDF(
            add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf"),
            self.obj_pos,
            [0, 0, 0, 1],
            useFixedBase=True,
        )


    def step(self, action):

        # scale and embed actions appropriately
        encoded_actions = self.encode_actions(action)
        scaled_actions = self.scale_actions(encoded_actions)

        self._env_step_counter += 1
        # set_trace()
        if self.user_mode == 'manual':
            self.robot.apply_action(
                scaled_actions,
                control_mode=self.control_mode,
                velocity_action_repeat=self._velocity_action_repeat,
                max_steps=self._max_blocking_pos_move_steps,
            )
        else:
            # Auto mode
            # Sample random position on the hemisphere
            
            # =============This is setup for sphere, just do it once!=============
            self.obj_obj_path = os.path.join(self.obj_path, "model.obj")
            mesh_original = utils_mesh._as_mesh(trimesh.load(self.obj_obj_path))
            # Process object vertices to match thee transformations on the urdf file
            vertices_wrld = utils_mesh.rotate_pointcloud(mesh_original.vertices, self.obj_rpy) * self.obj_scale + self.obj_pos
            mesh = trimesh.Trimesh(vertices=vertices_wrld, faces=mesh_original.faces)
            ray_hemisphere = utils_sample.get_ray_hemisphere(mesh)
            # ============= just do it once!=============


            self.robot.results_at_touch_wrld = None
            self.if_fault_collected_img = False
            hemisphere_random_pos, angles = utils_sample.sample_sphere(ray_hemisphere)
            # Move robot to random position on the hemisphere
            robot_sphere_wrld = mesh.bounding_box.centroid + np.array(hemisphere_random_pos)
            self.robot = utils_sample.robot_touch_spherical(self.robot, robot_sphere_wrld, self.obj_pos, angles)
            # Check on camera and store tactile images
            camera = self.robot.get_tactile_observation()
            check_on_camera = utils_sample.check_on_camera(camera)
            if not check_on_camera:
                #pb.removeBody(robot.robot_id)
                self.if_fault_collected_img = True
            
            # Filter points with information about contact, make sure there are at least {num_valid_points} valid ones
            contact_pointcloud = utils_raycasting.filter_point_cloud(self.robot.results_at_touch_wrld, self.obj_id)
            check_on_contact_pointcloud = utils_sample.check_on_contact_pointcloud(contact_pointcloud, self.num_valid_points)
            if not check_on_contact_pointcloud:
                print(f'Point cloud shape is too small: {contact_pointcloud.shape[0]} points')
                #pb.removeBody(robot.robot_id)
                self.if_fault_collected_img = True


        self._observation = self.get_observation()
        reward, done = self.get_step_data()



        return self._observation, reward, done, {}
    
    def get_extended_feature_array(self):
        # get sim info on TCP
        (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        (
            tcp_pos_worldframe,
            tcp_rpy_worldframe,
            _,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_worldframe()

        feature_array = np.array(
            [[
                *tcp_pos_workframe,
                *tcp_rpy_workframe,
            ],
            [
                *tcp_pos_worldframe,
                *tcp_rpy_worldframe,
            ]],
        )
        return feature_array
    
    def reset_task(self):
        """
        Randomise amount tip embedded into obj
        Reorientate obj
        """
        # reset the ur5 arm at the origin of the workframe with variation to the embed distance
                # Deactivate collision between robot and object. Raycasting to extract point cloud still works.
        pass

    def update_init_pose(self):
        """
        update the initial pose to be taken on reset, relative to the workframe
        """
        init_TCP_pos = [0, 0, 0]
        # init_TCP_pos = [0,0, 0]
        init_TCP_rpy = np.array([0.0, 0.0, 0.0])

        return init_TCP_pos, init_TCP_rpy

    def reset(self):
        """
        Reset the environment after an episode terminates.
        """

        # full reset pybullet sim to clear cache, this avoids silent bug where memory fills and visual
        # rendering fails, this is more prevelant when loading/removing larger files
        if self.reset_counter == self.reset_limit:
            self.full_reset()
        init_TCP_pos, init_TCP_rpy = self.update_init_pose()
        self.robot.reset(reset_TCP_pos=init_TCP_pos, reset_TCP_rpy=init_TCP_rpy)
        for link_idx in range(self._pb.getNumJoints(self.robot.robot_id)+1):
            self._pb.setCollisionFilterPair(self.robot.robot_id, self.obj_id, link_idx, -1, 0)
        self.reset_counter += 1
        self._env_step_counter = 0

        # update the workframe to a new position if embed dist randomisations are on
        # init_TCP_pos, init_TCP_rpy = self.update_init_pose()
        # self.robot.reset(reset_TCP_pos=init_TCP_pos, reset_TCP_rpy=init_TCP_rpy)

        # just to change variables to the reset pose incase needed before taking
        # a step
        self.get_step_data()

        # get the starting observation
        self._observation = self.get_observation()

        return self._observation

    def full_reset(self):
        self._pb.resetSimulation()
        self.load_environment()
        self.load_obj()
        self.robot.full_reset()
        self.reset_counter = 0

    def encode_actions(self, actions):
        """
        return actions as np.array in correct places for sending to ur5
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == "xy":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
        if self.movement_mode == "xyz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[2] = actions[2]
        if self.movement_mode == "xyRz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[5] = actions[2]
        if self.movement_mode == "xyzRz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[2] = actions[2]
            encoded_actions[5] = actions[3]
        if self.movement_mode == "xyzRxRyRz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[2] = actions[2]
            encoded_actions[3] = actions[3]
            encoded_actions[4] = actions[4]
            encoded_actions[5] = actions[5]
        return encoded_actions

    def get_step_data(self):

        # get the cur tip pos here for once per step
        (
            self.cur_tcp_pos_worldframe,
            self.cur_tcp_rpy_worldframe,
            self.cur_tcp_orn_worldframe,
            _,
            _,
        ) = self.robot.arm.get_current_TCP_pos_vel_worldframe()



        # if self.user_mode =='manual':
        if not self.if_fault_collected_img:
            if not self.robot.stop_at_touch:
                self.robot.stop_at_touch = True
                # self.if_collect_data = input("Are you sure to collect the tactile image?")
                # set_trace()
                # if self.if_collect_data:
                self.collected_number += 1
                tactile_image = self._observation['tactile']
                img_name = "image_" + str(self.collected_number) + ".png"
                cv2.imwrite(os.path.join(self.images_path, img_name), tactile_image)
                label_pose = self._observation['extended_feature']
                self.collected_3d_data_poses_workframe.append(label_pose[0])
                self.collected_3d_data_poses_worldframe.append(label_pose[1])

        # get rl info
        done = self.termination()
        reward = self.reward()

        return reward, done


    def termination(self):
        if self.collected_number == self.total_collected_number:
            np.save(os.path.join(self.data_path, 'poses_workframe.npy'), self.collected_3d_data_poses_workframe )
            np.save(os.path.join(self.data_path, 'poses_worldframe.npy'), self.collected_3d_data_poses_worldframe )
            return True
        else:
            return False



    def reward(self):

        return 1

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        # get sim info on TCP
        (
            tcp_pos_workframe,
            tcp_rpy_workframe,
            tcp_orn_workframe,
            tcp_lin_vel_workframe,
            tcp_ang_vel_workframe,
        ) = self.robot.arm.get_current_TCP_pos_vel_workframe()

        observation = np.hstack(
            [
                *tcp_pos_workframe,
                *tcp_rpy_workframe,
                *tcp_orn_workframe,
            ]
        )
        return observation

    def get_act_dim(self):
        if self.movement_mode == "xy":
            return 2
        if self.movement_mode == "xyz":
            return 3
        if self.movement_mode == "xyRz":
            return 3
        if self.movement_mode == "xyzRz":
            return 4
        if self.movement_mode == "xyzRxRyRz":
            return 6
        
def sample_sphere(r):
    """
    Uniform sampling on a hemisphere.
    Parameter:
        - r: radius
    Returns:
        - [x, y, z]: list of points in world frame
        - [phi, theta]: phi is horizontal (0, pi/2), theta is vertical (0, pi/2) 
    """
    phi = 2 * np.pi * np.random.uniform()
    theta = np.arccos(1 - 2 * np.random.uniform())
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    coords = [x, y, z]
    angles = [phi, theta]

    return coords, angles
