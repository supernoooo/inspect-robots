"""Isaac Lab direct task for bimanual YAM put-everything-in-box evaluation.

This module intentionally imports Isaac dependencies at module scope. The public
package imports it only after ``AppLauncher`` has started, as required by Isaac
Lab. The lightweight embodiment and contract modules remain importable anywhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from inspect_robots_isaacsim_2yam.contract import GRIPPER_OPEN_POSITION
from inspect_robots_isaacsim_2yam.embodiment import ISAAC_TASK_ID

_LEFT_ARM_NAMES = [f"left_joint{i}" for i in range(1, 7)]
_RIGHT_ARM_NAMES = [f"right_joint{i}" for i in range(1, 7)]
_LEFT_FINGER_NAMES = ["left_left_finger", "left_right_finger"]
_RIGHT_FINGER_NAMES = ["right_left_finger", "right_right_finger"]
# The original task places the robot at x=-0.65. Isaac's fixed MJCF root must
# stay at the USD origin, so every scene item is expressed in robot-base space.
_BOX_CENTER = (0.50, 0.0)
_SURFACE_Z = -0.01
_BOX_INNER_HALF = 0.09
_BOX_HEIGHT = 0.06
_BOX_WALL = 0.008


def _horizontal_aperture(hfov_degrees: float, focal_length: float = 24.0) -> float:
    return 2.0 * focal_length * math.tan(math.radians(hfov_degrees) / 2.0)


def _rigid_properties() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        disable_gravity=False,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=1,
    )


@configclass
class YamPutEverythingInBoxEnvCfg(DirectRLEnvCfg):
    """Configuration populated by :func:`make_env_cfg` after asset resolution."""

    decimation = 4
    episode_length_s = 400.0 / 30.0
    # DirectRLEnv.reset() forwards physics state before collecting the first
    # observation, but RTX sensors retain their previous render product unless
    # a reset re-render is requested.  Without this, episode N starts with the
    # exact final camera frame from episode N-1 while its joint state is already
    # reset.  One render refreshes every registered camera without advancing
    # physics or consuming an action step.
    num_rerenders_on_reset = 1
    action_space = 14
    observation_space = {  # noqa: RUF012 - Isaac configclass deep-copies config fields
        "top_cam": [360, 640, 3],
        "left_cam": [360, 640, 3],
        "right_cam": [360, 640, 3],
        "joint_pos": 14,
    }
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    robot: ArticulationCfg = MISSING
    duplo: RigidObjectCfg = MISSING
    ball: RigidObjectCfg = MISSING
    top_camera: TiledCameraCfg = MISSING
    left_camera: TiledCameraCfg = MISSING
    right_camera: TiledCameraCfg = MISSING
    spawn_noise = 0.02


def _robot_cfg(asset_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        articulation_root_prim_path="/bimanual_base/bimanual_base",
        spawn=sim_utils.MjcfFileCfg(
            asset_path=asset_path,
            # Let the imported MJCF expose one articulation root. Isaac Lab
            # fixes that root below; asking the importer to fix it creates a
            # second worldBody articulation in Isaac Sim 5.1.
            fix_base=False,
            import_sites=True,
            # The flattened bimanual MJCF intentionally reuses mesh files for
            # both arms. Isaac's instanceable mesh converter races when two
            # named mesh assets reference the same OBJ, so keep meshes embedded.
            make_instanceable=False,
            self_collision=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "left_joint[1-6]": 0.0,
                "right_joint[1-6]": 0.0,
                ".*_finger": GRIPPER_OPEN_POSITION,
            },
        ),
        actuators={
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_joint[1-3]"],
                effort_limit_sim=28.0,
                stiffness=40.0,
                damping=2.5,
            ),
            "elbows": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_joint4"],
                effort_limit_sim=10.0,
                stiffness=20.0,
                damping=0.5,
            ),
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_joint[5-6]"],
                effort_limit_sim=10.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=[".*_finger"],
                effort_limit_sim=40.0,
                stiffness=2000.0,
                damping=40.0,
            ),
        },
    )


def _object_cfg(prim_name: str, *, position: tuple[float, float, float]) -> RigidObjectCfg:
    common = {
        "rigid_props": _rigid_properties(),
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.1),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    }
    if prim_name == "duplo":
        spawn = sim_utils.CuboidCfg(
            size=(0.096, 0.032, 0.038),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.08, 0.04)),
            **common,
        )
    else:
        spawn = sim_utils.SphereCfg(
            radius=0.0335,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.83, 0.22)),
            **common,
        )
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def _camera_cfg(
    prim_path: str,
    *,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    hfov_degrees: float,
    height: int,
    width: int,
) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=prim_path,
        # MolmoAct2's source poses are SAPIEN poses (+X forward, +Z up), which
        # is Isaac Lab's ``world`` camera convention. Isaac then converts them
        # to the USD/OpenGL camera frame (-Z forward, +Y up).
        offset=TiledCameraCfg.OffsetCfg(pos=position, rot=rotation, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=_horizontal_aperture(hfov_degrees),
            clipping_range=(0.01, 10.0),
        ),
        height=height,
        width=width,
    )


def make_env_cfg(
    *,
    asset_path: str,
    device: str,
    height: int,
    width: int,
    control_hz: float,
    spawn_noise: float,
) -> YamPutEverythingInBoxEnvCfg:
    """Build one task configuration with runtime paths and camera dimensions."""
    cfg = YamPutEverythingInBoxEnvCfg()
    cfg.robot = _robot_cfg(asset_path)
    # Local primitives avoid a runtime dependency on Isaac's remote Nucleus
    # asset server while retaining the task's dimensions, colors, and physics.
    cfg.duplo = _object_cfg("duplo", position=(0.35, 0.22, _SURFACE_Z + 0.019))
    cfg.ball = _object_cfg("ball", position=(0.35, -0.22, _SURFACE_Z + 0.0335))
    cfg.top_camera = _camera_cfg(
        "/World/envs/env_.*/top_cam",
        position=(0.15, 0.0, 0.80),
        rotation=(0.7660444431, 0.0, 0.6427876097, 0.0),
        hfov_degrees=69.4,
        height=height,
        width=width,
    )
    wrist_rotation = (0.6123724292, -0.3535533915, -0.3535533967, -0.6123724381)
    cfg.left_camera = _camera_cfg(
        "/World/envs/env_.*/Robot/bimanual_base/left_link_6/left_cam",
        position=(0.0, 0.09, 0.06),
        rotation=wrist_rotation,
        hfov_degrees=87.0,
        height=height,
        width=width,
    )
    cfg.right_camera = _camera_cfg(
        "/World/envs/env_.*/Robot/bimanual_base/right_link_6/right_cam",
        position=(0.0, 0.09, 0.06),
        rotation=wrist_rotation,
        hfov_degrees=87.0,
        height=height,
        width=width,
    )
    cfg.observation_space = {
        "top_cam": [height, width, 3],
        "left_cam": [height, width, 3],
        "right_cam": [height, width, 3],
        "joint_pos": 14,
    }
    cfg.sim.device = device
    cfg.sim.dt = 1.0 / (control_hz * cfg.decimation)
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = 400.0 / control_hz
    cfg.spawn_noise = spawn_noise
    return cfg


def register_env() -> None:
    """Register the bundled Gymnasium task idempotently."""
    if ISAAC_TASK_ID not in gym.registry:
        gym.register(
            id=ISAAC_TASK_ID,
            entry_point="inspect_robots_isaacsim_2yam.isaac_env:YamPutEverythingInBoxEnv",
            disable_env_checker=True,
        )


class YamPutEverythingInBoxEnv(DirectRLEnv):
    """Direct Isaac Lab environment matching MolmoAct2's YAM simulation task."""

    cfg: YamPutEverythingInBoxEnvCfg

    def __init__(
        self, cfg: YamPutEverythingInBoxEnvCfg, render_mode: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        self._left_arm_ids, _ = self._robot.find_joints(_LEFT_ARM_NAMES, preserve_order=True)
        self._right_arm_ids, _ = self._robot.find_joints(_RIGHT_ARM_NAMES, preserve_order=True)
        self._left_finger_ids, _ = self._robot.find_joints(_LEFT_FINGER_NAMES, preserve_order=True)
        self._right_finger_ids, _ = self._robot.find_joints(
            _RIGHT_FINGER_NAMES, preserve_order=True
        )
        counts = tuple(
            len(ids)
            for ids in (
                self._left_arm_ids,
                self._right_arm_ids,
                self._left_finger_ids,
                self._right_finger_ids,
            )
        )
        if counts != (6, 6, 2, 2):
            raise RuntimeError(f"YAM MJCF joint layout mismatch: found group sizes {counts}")
        self._joint_targets = self._robot.data.default_joint_pos.clone()
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self._duplo = RigidObject(self.cfg.duplo)
        self._ball = RigidObject(self.cfg.ball)
        self._top_camera = TiledCamera(self.cfg.top_camera)
        self._left_camera = TiledCamera(self.cfg.left_camera)
        self._right_camera = TiledCamera(self.cfg.right_camera)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self._spawn_table_and_box()
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["duplo"] = self._duplo
        self.scene.rigid_objects["ball"] = self._ball
        self.scene.sensors["top_cam"] = self._top_camera
        self.scene.sensors["left_cam"] = self._left_camera
        self.scene.sensors["right_cam"] = self._right_camera
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85))
        light.func("/World/Light", light)

    def _spawn_table_and_box(self) -> None:
        table = sim_utils.CuboidCfg(
            size=(1.2, 0.9, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.36, 0.28)),
        )
        table.func("/World/envs/env_.*/Table", table, translation=(0.40, 0.0, _SURFACE_Z - 0.025))
        # Child spawners can expand an environment regex only after the common
        # parent exists in the template environment.
        sim_utils.create_prim("/World/envs/env_0/Box", "Xform")
        brown = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.18))
        outer = _BOX_INNER_HALF + _BOX_WALL
        parts = (
            ("floor", (2 * outer, 2 * outer, _BOX_WALL), (0.0, 0.0, _BOX_WALL / 2)),
            (
                "wall_x_pos",
                (_BOX_WALL, 2 * outer, _BOX_HEIGHT),
                (_BOX_INNER_HALF + _BOX_WALL / 2, 0.0, _BOX_WALL + _BOX_HEIGHT / 2),
            ),
            (
                "wall_x_neg",
                (_BOX_WALL, 2 * outer, _BOX_HEIGHT),
                (-_BOX_INNER_HALF - _BOX_WALL / 2, 0.0, _BOX_WALL + _BOX_HEIGHT / 2),
            ),
            (
                "wall_y_pos",
                (2 * _BOX_INNER_HALF, _BOX_WALL, _BOX_HEIGHT),
                (0.0, _BOX_INNER_HALF + _BOX_WALL / 2, _BOX_WALL + _BOX_HEIGHT / 2),
            ),
            (
                "wall_y_neg",
                (2 * _BOX_INNER_HALF, _BOX_WALL, _BOX_HEIGHT),
                (0.0, -_BOX_INNER_HALF - _BOX_WALL / 2, _BOX_WALL + _BOX_HEIGHT / 2),
            ),
        )
        for name, size, offset in parts:
            box_part = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=brown,
            )
            box_part.func(
                f"/World/envs/env_.*/Box/{name}",
                box_part,
                translation=(
                    _BOX_CENTER[0] + offset[0],
                    _BOX_CENTER[1] + offset[1],
                    _SURFACE_Z + offset[2],
                ),
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        limits = self._robot.data.soft_joint_pos_limits
        self._joint_targets[:] = self._robot.data.joint_pos
        self._joint_targets[:, self._left_arm_ids] = self.actions[:, :6]
        self._joint_targets[:, self._right_arm_ids] = self.actions[:, 7:13]
        left_gripper = GRIPPER_OPEN_POSITION * self.actions[:, 6].clamp(0.0, 1.0)
        right_gripper = GRIPPER_OPEN_POSITION * self.actions[:, 13].clamp(0.0, 1.0)
        self._joint_targets[:, self._left_finger_ids] = left_gripper.unsqueeze(1)
        self._joint_targets[:, self._right_finger_ids] = right_gripper.unsqueeze(1)
        self._joint_targets[:] = torch.clamp(self._joint_targets, limits[..., 0], limits[..., 1])

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        joint_pos = self._robot.data.joint_pos
        left_gripper = (-joint_pos[:, self._left_finger_ids[0]] / abs(GRIPPER_OPEN_POSITION)).clamp(
            0.0, 1.0
        )
        right_gripper = (
            -joint_pos[:, self._right_finger_ids[0]] / abs(GRIPPER_OPEN_POSITION)
        ).clamp(0.0, 1.0)
        state = torch.cat(
            (
                joint_pos[:, self._left_arm_ids],
                left_gripper.unsqueeze(1),
                joint_pos[:, self._right_arm_ids],
                right_gripper.unsqueeze(1),
            ),
            dim=1,
        )
        return {
            "policy": {
                "top_cam": self._top_camera.data.output["rgb"].clone()[..., :3],
                "left_cam": self._left_camera.data.output["rgb"].clone()[..., :3],
                "right_cam": self._right_camera.data.output["rgb"].clone()[..., :3],
                "joint_pos": state,
            }
        }

    def _inside_box(self, obj: RigidObject) -> torch.Tensor:
        position = obj.data.root_pos_w - self.scene.env_origins
        return (
            (torch.abs(position[:, 0] - _BOX_CENTER[0]) < _BOX_INNER_HALF)
            & (torch.abs(position[:, 1] - _BOX_CENTER[1]) < _BOX_INNER_HALF)
            & (position[:, 2] > _SURFACE_Z + _BOX_WALL - 0.01)
            & (position[:, 2] < _SURFACE_Z + _BOX_WALL + _BOX_HEIGHT + 0.05)
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._success = self._inside_box(self._duplo) & self._inside_box(self._ball)
        self.extras["success"] = self._success
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self._success, truncated

    def _get_rewards(self) -> torch.Tensor:
        return (self._inside_box(self._duplo).float() + self._inside_box(self._ball).float()) / 2.0

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_pos
        self._reset_object(self._duplo, env_ids, x=0.35, y=0.22, z=_SURFACE_Z + 0.019)
        self._reset_object(self._ball, env_ids, x=0.35, y=-0.22, z=_SURFACE_Z + 0.0335)

    def _reset_object(
        self,
        obj: RigidObject,
        env_ids: Sequence[int],
        *,
        x: float,
        y: float,
        z: float,
    ) -> None:
        state = obj.data.default_root_state[env_ids].clone()
        count = len(env_ids)
        noise = (2.0 * torch.rand((count, 2), device=self.device) - 1.0) * self.cfg.spawn_noise
        state[:, 0] = x + noise[:, 0]
        state[:, 1] = y + noise[:, 1]
        state[:, 2] = z
        state[:, :3] += self.scene.env_origins[env_ids]
        state[:, 7:] = 0.0
        obj.write_root_pose_to_sim(state[:, :7], env_ids)
        obj.write_root_velocity_to_sim(state[:, 7:], env_ids)
