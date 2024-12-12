
from typing import Union, Any, Dict, Optional
import numpy as np
import sapien
import torch


from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig, SceneConfig

from robots.simple_fetch import SimpleFetch


@register_env("PlaySoccer-v1", max_episode_steps=100)
class SoccerPlayEnv(BaseEnv):
    agent: Union[SimpleFetch]
    
    def __init__(self, *args, robot_uids="simp_fetch", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise=robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=100,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            ),
            scene_config=SceneConfig(
                bounce_threshold=0
            )
        )
    
    @property
    def _default_sensor_configs(self):
        return []
    
    def _load_agent(self, options: dict):
        return super()._load_agent(options, sapien.Pose(p=[1, 0, 0]))
    
    def _load_scene(self, options):
        
        # Create the Ground
        self.ground = build_ground(self.scene)
        sapien.set_log_level("warn")
        
        from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
        self.ground.set_collision_group_bit(
            group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
        )
        
        
        
        # Create the walls
        self._build_walls(wall_length=10)
        
        # Create a Ball Object
        self._build_balls(1)
        
        # Get force contact links
        self.active_links = [
            link for link in self.agent.robot.get_links() if "dummy" not in link.name
        ]
        
        # print("Note: These are the active links -->", [link.name for link in self.active_links])
        
        self.force_sensor_links = [
            link.name for link in self.active_links if "force" in link.name
        ]
        # print("Note: These are the force sensor links -->", self.force_sensor_links)
        
     
    def _build_balls(self, num_balls=1):
        builder = self.scene.create_actor_builder()
        
        self.balls = []
        for ball_num in range(num_balls):
            radius = randomization.uniform(0.1, 0.3, (1,))
            ball_color = randomization.uniform(0, 1, (3,))
                    
            #TODO: Make this randomized        
            
            builder.add_sphere_collision(pose=sapien.Pose([0, 0, 0]), radius=radius)
            builder.add_sphere_visual(pose=sapien.Pose([0, 0, 0]), radius=radius, material=ball_color)
            builder.initial_pose = sapien.Pose(p=[0, 0, 0]) #TODO: Change this plz
            ball = builder.build(name=f"ball{ball_num}")        
            ball.set_linear_damping(0)
            
            self.balls.append(ball)
      
    def _build_walls(self, wall_thickness=0.05, wall_height=0.4, wall_length = 2.0):

        builder = self.scene.create_actor_builder()
        
        # Add front wall
        builder.add_box_collision(
            pose=sapien.Pose([0, wall_length / 2 + wall_thickness / 2, wall_height / 2]),
            half_size=[wall_length / 2, wall_thickness / 2, wall_height / 2],
        )
        builder.add_box_visual(
            pose=sapien.Pose([0, wall_length / 2 + wall_thickness / 2, wall_height / 2]),
            half_size=[wall_length / 2, wall_thickness / 2, wall_height / 2],
        )

        # Add back wall
        builder.add_box_collision(
            pose=sapien.Pose([0, -wall_length / 2 - wall_thickness / 2, wall_height / 2]),
            half_size=[wall_length / 2, wall_thickness / 2, wall_height / 2],
        )
        builder.add_box_visual(
            pose=sapien.Pose([0, -wall_length / 2 - wall_thickness / 2, wall_height / 2]),
            half_size=[wall_length / 2, wall_thickness / 2, wall_height / 2],
        )

        # Add left wall
        builder.add_box_collision(
            pose=sapien.Pose([-wall_length / 2 - wall_thickness / 2, 0, wall_height / 2]),
            half_size=[wall_thickness / 2, wall_length / 2, wall_height / 2],
        )
        builder.add_box_visual(
            pose=sapien.Pose([-wall_length / 2 - wall_thickness / 2, 0, wall_height / 2]),
            half_size=[wall_thickness / 2, wall_length / 2, wall_height / 2],
        )

        # Add right wall
        builder.add_box_collision(
            pose=sapien.Pose([wall_length / 2 + wall_thickness / 2, 0, wall_height / 2]),
            half_size=[wall_thickness / 2, wall_length / 2, wall_height / 2],
        )
        builder.add_box_visual(
            pose=sapien.Pose([wall_length / 2 + wall_thickness / 2, 0, wall_height / 2]),
            half_size=[wall_thickness / 2, wall_length / 2, wall_height / 2],
        )

        builder.initial_pose = sapien.Pose()  # Center the walls around the origin
        self.walls = builder.build_static(name="walls")
    
    @property
    def contact_forces(self):
        """Returns log1p of force on contact-sensor links"""
        force_vecs = torch.stack(
            [self.agent.robot.get_net_contact_forces([link]) for link in self.force_sensor_links],
            dim=1
        )
        
        force_mags = torch.linalg.norm(force_vecs, dim=-1).view(-1, len(self.force_sensor_links)) # (b, len(contact_sensors))
        return torch.log1p(force_mags)
        
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0, -4, 5], target=[0, 0, 0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
        
    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)


            if self.robot_uids == "simp_fetch":
                qpos = torch.zeros(b, 5)
                dist = randomization.uniform(2.0, 2.5, size=(b,))
                theta = randomization.uniform(0.9 * torch.pi, 1.1 * torch.pi, size=(b,))
                xy = torch.zeros((b, 2))
                xy[:, 0] += torch.cos(theta) * dist
                xy[:, 1] += torch.sin(theta) * dist
                qpos[:, :2] = xy
                ori = (theta - torch.pi) + randomization.uniform(-0.05 * torch.pi, 0.05 * torch.pi, size=(b,))
                qpos[:, 2] = ori
                qpos[:, 3] = randomization.uniform(-1.0, 1.0, (b,)) # randomize camera pan
                qpos[:, 4] = randomization.uniform(-0.7, 1.2, (b,)) # randomize camera tilt
                
                
                self.agent.robot.set_qpos(qpos)
                self.agent.robot.set_pose(sapien.Pose())
                
                # Initialize the Balls
                direct_at_agent = True
                for ball in self.balls:
                    # Randomize the Ball Positions
                    poses = randomization.uniform(-3,3, (b, 7))
                    poses[:, 0] = torch.abs(poses[:, 0])
                    poses[:, 2] = 0.1
                    poses[:, 4:] = 0
                    poses[:, 3] = 1
                    ball.pose = poses
                    
                    
                    # Randomize the ball velocity
                    min_vel = 0.3
                    max_vel = 2
                    ball_vels = randomization.uniform(min_vel, max_vel, (b, 1))

                    ball_xys = ball.pose.raw_pose[:,0:2]
                    
                    if direct_at_agent:
                        noise_amt = 0.1
                        noise = randomization.uniform(noise_amt, noise_amt, (b, 2))
                        vec_to_agent = (xy - ball_xys) + noise
                        unit_vec_to_agent = vec_to_agent / (torch.linalg.norm(vec_to_agent, dim=1, keepdim=True) + 1e-8)
                    
                        unit_vec_to_agent3 = torch.zeros((b, 3))
                        unit_vec_to_agent3[:, :2] = unit_vec_to_agent
                        lin_vel = (unit_vec_to_agent3)*ball_vels
                        
                        direct_at_agent = False
                    else:
                        lin_vel = randomization.uniform(-max_vel, max_vel, (b, 3))
                        lin_vel[:, 2] = 0
                        
                    ball.set_linear_velocity(lin_vel)
            
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()
            
    
    # @property
    # def _after_control_step(self):
    #     if self.gpu_sim_enabled:
    #         self.scene.px.gpu_apply_rigid_dynamic_data()
    #         self.scene.px.gpu_fetch_rigid_dynamic_data()
            
    def _after_control_step(self):     
        return super()._after_control_step()
    
    
    # @property
    # def evaluate(self): # TODO: maybe do better than this lol
    #     print("We are trying to evaluate I think")
    #     return dict(
    #         success=False
    #     )
        
    def _get_obs_extra(self, info): #TODO: right now, it breaks if this is empty for some reason....
        
        robot_position = self.agent.robot.get_pose().p
        # ball_position = self.ball.pose.p
                
        distance = torch.norm(robot_position-robot_position, dim=1)
        
        # ball_position = self.ball.pose.p
        
        # random_obs = torch.zeros_like(ball_position, device=ball_position.device) + 43
        obs = dict(
            dist=torch.zeros_like(distance),
            contact_forces=self.contact_forces
        )
        return obs
    
    def compute_dense_reward(self, obs, action, info):
        robot_position = self.agent.robot.get_pose().p
        # ball_position = self.ball.pose.p
                
        distance = torch.norm(robot_position-robot_position, dim=1) # TODO: This is the distance to thing

        meaningless_reward = torch.ones_like(distance, device=action.device) #TODO: Make this better please
        return meaningless_reward
    
    def compute_normalized_dense_reward(self, obs, action, info):
        
        return self.compute_dense_reward(obs, action, info) / 10 #TODO: This needs to be updated lol
    
    # def compute_sparse_reward(self, obs, action, info):
    #     return super().compute_sparse_reward(obs, action, info)
                
                
            
        
        
