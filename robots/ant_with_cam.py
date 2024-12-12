from copy import deepcopy
from typing import Dict, Tuple, Any, Optional, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array
from transforms3d.euler import euler2quat

FETCH_WHEELS_COLLISION_BIT = 30
"""Collision bit of the fetch robot wheel links"""
FETCH_BASE_COLLISION_BIT = 31
"""Collision bit of the fetch base"""

import os
current_directory = os.path.dirname(os.path.abspath(__file__))


@register_agent()
class CamAnt(BaseAgent):
    uid = "cam_ant"
    mjcf_path = os.path.join(current_directory, f"./ant_cam.xml")
    # urdf_config = dict(
    # )
    fix_root_link = False
    
    
    keyframes = dict(
        rest=Keyframe(
            # qpos=np.array([0, 0, 0, 0, 1, -1, -1, 1]),
            qpos=np.array([0, 0, 0, 0, 0, 0, 1, -1, -1, 1]),
            pose=sapien.Pose(p=[0, 0, -0.175], q=euler2quat(0, 0, np.pi / 2)),
        )
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="ant_head",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="camera_body", #TODO: Change this
            )
        ]

    def __init__(self, *args, **kwargs):
        # self.body_joint_names = [
        #     "head_pan_joint",
        #     "head_tilt_joint",
        # ]
        # self.body_stiffness = 1e3
        # self.body_damping = 1e2
        # self.body_force_limit = 10000

        # self.base_joint_names = [
        #     "root_x_axis_joint",
        #     "root_y_axis_joint",
        #     "root_z_rotation_joint",
        # ]

        super().__init__(*args, **kwargs)
        
    def _load_articulation(
        self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None
    ):
        """
        Load the robot articulation
        """
        loader = self.scene.create_mjcf_loader()
        asset_path = str(self.mjcf_path)

        loader.name = self.uid

        builder = loader.parse(asset_path)["articulation_builders"][0]
        builder.initial_pose = initial_pose
        self.robot = builder.build()
        assert self.robot is not None, f"Fail to load URDF/MJCF from {asset_path}"
        self.robot_link_ids = [link.name for link in self.robot.get_links()]

    @property
    def _controller_configs(self):
        print("These are the joint names", [joint.name for joint in self.robot.get_active_joints()])
        
        body = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.get_active_joints()],
            lower=-1,
            upper=1,
            damping=1e2,
            stiffness=1e3,
            use_delta=True,
        )
        return deepcopy_dict(
            dict(
                pd_joint_delta_pos=dict(
                    body=body,
                    balance_passive_force=False,
                ),
            )
        )
    # def _after_init(self):
    #     self.base_link: Link = sapien_utils.get_obj_by_name(
    #         self.robot.get_links(), "base_link"
    #     )
    #     self.l_wheel_link: Link = self.robot.links_map["l_wheel_link"]
    #     self.r_wheel_link: Link = self.robot.links_map["r_wheel_link"]
    #     for link in [self.l_wheel_link, self.r_wheel_link]:
    #         link.set_collision_group_bit(
    #             group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
    #         )
    #     self.base_link.set_collision_group_bit(
    #         group=2, bit_idx=FETCH_BASE_COLLISION_BIT, bit=1
    #     )

    #     self.head_camera_link: Link = sapien_utils.get_obj_by_name(
    #         self.robot.get_links(), "head_camera_link"
    #     )

    #     self.queries: Dict[
    #         str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
    #     ] = dict()


    # def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
    #     body_qvel = self.robot.get_qvel()[..., 3:-2]
    #     base_qvel = self.robot.get_qvel()[..., :3]
    #     return torch.all(body_qvel <= threshold, dim=1) & torch.all(
    #         base_qvel <= base_threshold, dim=1
    #     )
