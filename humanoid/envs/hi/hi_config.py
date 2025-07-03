# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class HiCfg(LeggedRobotCfg):
    """
    Configuration class for the high-torque Pi robot.
    """

    class env(LeggedRobotCfg.env):
        # num_observations与num_privileged_obs决定着神经网络中某些行列式的大小，
        # 必须要让他们的参数合适对应才能进行矩阵乘法，才能让神经网络跑的下去。
        # change the observation dim
        frame_stack = 15                    # 策略网络的帧堆叠数量，存储历史观测数据用于时序决策
        c_frame_stack = 3                   # 评论家网络的帧堆叠数量，为价值函数提供较短的历史信息
        num_single_obs = 86 #47             # 单个时间步的观测维度（包含命令、关节状态、动作等）
        num_observations = int(frame_stack * (num_single_obs))  # 评论家网络输入维度
        single_num_privileged_obs = 125 #73 # 单个时间步的特权观测维度（包含额外状态信息如接触力、摩擦力等）
        num_privileged_obs = int(c_frame_stack * (single_num_privileged_obs))
        num_actions = 25 #12                # 动作空间维度，对应25个关节的扭矩控制
        num_envs = 4096                     # 并行环境的数量，同时运行4096个机器人环境进行训练
        episode_length_s = 12  # episode length in seconds
        use_ref_actions = False             # 是否使用参考动作，False表示不使用参考轨迹

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/robot_urdf-hi_25dof_250607/urdf/hi_25dof_250607_rl.urdf"

        name = "Hi"
        foot_name = "ankle_roll"
        knee_name = "calf"

        terminate_after_contacts_on = ["base_link"]
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0

    class noise:
        add_noise = True
        noise_level = 0.6  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        # rot = [0., 0.27154693695611287, 0., 0.962425197628238]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "r_hip_pitch_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.0,
            "r_calf_joint": 0.0,
            "r_ankle_pitch_joint": 0.0,
            "r_ankle_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "l_calf_joint": 0.0,
            "l_ankle_pitch_joint": 0.0,
            "l_ankle_roll_joint": 0.0,

            # 手臂关节
            "l_shoulder_pitch_joint": 0.0,
            "l_shoulder_roll_joint": 0.0,
            "l_upper_arm_joint": 0.0,
            "l_elbow_joint": 0.0,
            "l_wrist_joint": 0.0,
            "r_shoulder_pitch_joint": 0.0,
            "r_shoulder_roll_joint": 0.0,
            "r_upper_arm_joint": 0.0,
            "r_elbow_joint": 0.0,
            "r_wrist_joint": 0.0,
            
            # 头部关节
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
            
            # 腰部关节
            "waist_joint": 0.0
        }

    class control(LeggedRobotCfg.control):
      # PD Drive parameters:
        stiffness = {
            # 腿部关节 (12个)
            "r_hip_pitch_joint": 40.0,
            "r_hip_roll_joint": 20.0,
            "r_thigh_joint": 20.0,
            "r_calf_joint": 40.0,
            "r_ankle_pitch_joint": 40.0,
            "r_ankle_roll_joint": 20.0,
            "l_hip_pitch_joint": 40.0,
            "l_hip_roll_joint": 20.0,
            "l_thigh_joint": 20.0,
            "l_calf_joint": 40.0,
            "l_ankle_pitch_joint": 40.0,
            "l_ankle_roll_joint": 20.0,
            
            # 手臂关节 (10个)
            "r_shoulder_pitch_joint": 30.0,
            "r_shoulder_roll_joint": 15.0,
            "r_upper_arm_joint": 20.0,
            "r_elbow_joint": 30.0,
            "r_wrist_joint": 10.0,
            "l_shoulder_pitch_joint": 30.0,
            "l_shoulder_roll_joint": 15.0,
            "l_upper_arm_joint": 20.0,
            "l_elbow_joint": 30.0,
            "l_wrist_joint": 10.0,
            
            # 头部关节 (2个)
            "head_yaw_joint": 5.0,
            "head_pitch_joint": 5.0,
            
            # 腰部关节 (1个)
            "waist_joint": 15.0
        }

        damping = {
            # 腿部关节
            "r_hip_pitch_joint": 0.6,
            "r_hip_roll_joint": 0.4,
            "r_thigh_joint": 0.4,
            "r_calf_joint": 0.6,
            "r_ankle_pitch_joint": 0.6,
            "r_ankle_roll_joint": 0.4,
            "l_hip_pitch_joint": 0.6,
            "l_hip_roll_joint": 0.4,
            "l_thigh_joint": 0.4,
            "l_calf_joint": 0.6,
            "l_ankle_pitch_joint": 0.6,
            "l_ankle_roll_joint": 0.4,
            
            # 手臂关节
            "r_shoulder_pitch_joint": 0.5,
            "r_shoulder_roll_joint": 0.3,
            "r_upper_arm_joint": 0.3,
            "r_elbow_joint": 0.5,
            "r_wrist_joint": 0.2,
            "l_shoulder_pitch_joint": 0.5,
            "l_shoulder_roll_joint": 0.3,
            "l_upper_arm_joint": 0.3,
            "l_elbow_joint": 0.5,
            "l_wrist_joint": 0.2,
            
            # 头部关节
            "head_yaw_joint": 0.1,
            "head_pitch_joint": 0.1,
            
            # 腰部关节
            "waist_joint": 0.3
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        # decimation = 10  # 100hz
        decimation = 20  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 30
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_Hirs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.5
        min_dist = 0.15
        max_dist = 0.2
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.08  # rad
        target_feet_height = 0.02  # m 是否要更高
        cycle_time = 0.4  # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma_ang = 0.1
        tracking_sigma_lin = 0.1
        max_contact_force = 100  # forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 1.6  # 1.6
            feet_clearance = 5.0
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.0
            foot_slip = -0.05
            feet_distance = 0.16  # 0.2
            knee_distance = 0.16  # 0.2
            # contact
            feet_contact_forces = -0.001
            # vel tracking
            tracking_lin_vel = 10
            tracking_ang_vel = 20
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed =0.05
            track_vel_hard = 0.2
            # base pos
            default_hip_roll_joint_pos = 4
            default_thigh_joint_pos = 1.0
            default_ankle_roll_pos = 0.5
            orientation = 0.5
            base_height = 0.5
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-5
            dof_acc = -1e-8
            collision = -1.0

            termination = 1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0


class HiCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 10001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "Hi_ppo"
        run_name = "v2"
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and checkpoint
