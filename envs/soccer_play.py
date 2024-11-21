
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
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from robots.simple_fetch import SimpleFetch



@register_env("PlaySoccer-v1", max_episode_steps=1000)
class SoccerPlayEnv(BaseEnv):
    agent: Union[SimpleFetch]
    
    def __init__(self, *args, robot_uids="simp_fetch", robot_init_qpos_noise=0.02, **kwargs):
        print("Here we are")
        self.robot_init_qpos_noise=robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=100,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            )
        )
    
    @property
    def _default_sensor_configs(self):
        return []
    
    def _load_agent(self, options: dict):
        return super()._load_agent(options, sapien.Pose(p=[1, 0, 0]))
    
    def _load_scene(self, options):
        self.ground = build_ground(self.scene)
        sapien.set_log_level("warn")
        
        from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
        
        self.ground.set_collision_group_bit(
            group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
        )
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-1.8, -1.3, 1.8], target=[0, 0, 0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
        
    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)


            if self.robot_uids == "simp_fetch":
                qpos = torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
                qpos = qpos.repeat(b).reshape(b, -1)
                dist = randomization.uniform(1.6, 1.8, size=(b,))
                theta = randomization.uniform(0.9 * torch.pi, 1.1 * torch.pi, size=(b,))
                xy = torch.zeros((b, 2))
                xy[:, 0] += torch.cos(theta) * dist
                xy[:, 1] += torch.sin(theta) * dist
                qpos[:, :2] = xy # TODO, Idk if this is right? 
                noise_ori = randomization.uniform(
                    -0.05 * torch.pi, 0.05 * torch.pi, size=(b,)
                )
                ori = (theta - torch.pi) + noise_ori
                qpos[:, 2] = ori
                self.agent.robot.set_qpos(qpos)
                self.agent.robot.set_pose(sapien.Pose())
            
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
        
    # def _get_obs_extra(self, info):
    #     return super()._get_obs_extra(info)
    
    # def compute_dense_reward(self, obs, action, info):
    #     return super().compute_dense_reward(obs, action, info)
    
    # def compute_sparse_reward(self, obs, action, info):
    #     return super().compute_sparse_reward(obs, action, info)
                
                
            
        
        
