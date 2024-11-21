from copy import deepcopy
from typing import Dict, Tuple

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

FETCH_WHEELS_COLLISION_BIT = 30
"""Collision bit of the fetch robot wheel links"""
FETCH_BASE_COLLISION_BIT = 31
"""Collision bit of the fetch base"""

import os
current_directory = os.path.dirname(os.path.abspath(__file__))


@register_agent()
class SimpleFetch(BaseAgent):
    uid = "simp_fetch"
    urdf_path = os.path.join(current_directory, f"./simplified_fetch.urdf")
    urdf_config = dict(
    )

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            qpos=np.array(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ),
        )
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="fetch_head",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="head_camera_link",
            ),
            # CameraConfig(
            #     uid="fetch_hand",
            #     pose=Pose.create_from_pq([-0.1, 0, 0.1], [1, 0, 0, 0]),
            #     width=128,
            #     height=128,
            #     fov=2,
            #     near=0.01,
            #     far=100,
            #     entity_uid="gripper_link",
            # ),
        ]

    def __init__(self, *args, **kwargs):
        self.body_joint_names = [
            "head_pan_joint",
            "head_tilt_joint",
        ]
        self.body_stiffness = 1e3
        self.body_damping = 1e2
        self.body_force_limit = 10000

        self.base_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
        ]

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):

        # -------------------------------------------------------------------------- #
        # Body
        # -------------------------------------------------------------------------- #
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            -0.3,
            0.3,
            self.body_stiffness,
            self.body_damping,
            self.body_force_limit,
            use_delta=True,
        )

        # useful to keep body unmoving from passed position
        stiff_body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            None,
            None,
            1e5,
            1e5,
            1e5,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        base_pd_joint_vel = PDBaseForwardVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -3.14],
            upper=[1, 3.14],
            damping=1000,
            force_limit=500,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos=dict(
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos_vel=dict(
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_vel=dict(
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_stiff_body=dict(
                body=stiff_body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link"
        )
        self.l_wheel_link: Link = self.robot.links_map["l_wheel_link"]
        self.r_wheel_link: Link = self.robot.links_map["r_wheel_link"]
        for link in [self.l_wheel_link, self.r_wheel_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        self.base_link.set_collision_group_bit(
            group=2, bit_idx=FETCH_BASE_COLLISION_BIT, bit=1
        )

        self.head_camera_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "head_camera_link"
        )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()


    def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
        body_qvel = self.robot.get_qvel()[..., 3:-2]
        base_qvel = self.robot.get_qvel()[..., :3]
        return torch.all(body_qvel <= threshold, dim=1) & torch.all(
            base_qvel <= base_threshold, dim=1
        )
