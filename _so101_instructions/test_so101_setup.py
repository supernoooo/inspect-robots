"""CLI rig and upstream home lifecycle checks with fake motors; never open hardware."""

import signal
import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from inspect_robots import Action, Scene
from inspect_robots.errors import ConfigError
from inspect_robots_so101 import SOArmConfig, SOArmEmbodiment, packing

from _run_so101 import configured_so101, rig_config
from _so101_motion import CLIArm, HomeSession, _connected_driver


class FakeDriver:
    def __init__(self, config, pose):
        self.config = config
        self.pose = np.array(pose, dtype=float)
        self.commands = []
        self.disconnected = False
        self.fail_send = False
        self.stuck_joint = None
        self.command_limits = []

    def get_observation(self):
        return {**packing.to_action_dict(self.pose), **{
            name: np.zeros((self.config.cam_height, self.config.cam_width, 3), dtype=np.uint8)
            for name in self.config.cameras}}

    def send_action(self, action):
        if self.fail_send:
            raise OSError('motor connection failed')
        wanted = packing.from_obs_dict(action)
        limit = self.config.max_relative_target
        accepted = wanted if limit is None else self.pose + np.clip(wanted - self.pose, -limit, limit)
        old_pose = self.pose.copy()
        self.pose = accepted.copy()
        if self.stuck_joint is not None:
            self.pose[self.stuck_joint] = old_pose[self.stuck_joint]
        self.commands.append(accepted.copy())
        self.command_limits.append(limit)
        return packing.to_action_dict(accepted)

    def disconnect(self):
        self.disconnected = True


class RigSetupTests(unittest.TestCase):
    def rig(self, *, home_pose=None, pose=(0, 0, 0, 0, 0, 50), **options):
        cfg = SOArmConfig(max_relative_target=None,
                          settle_tolerance=0.1, control_hz=10,
                          cameras=('front', 'head'), cam_width=16, cam_height=16)
        driver = FakeDriver(cfg, pose)
        ticks = [0.0]
        sleeps = []
        ready = []

        def sleep(duration):
            sleeps.append(duration)
            ticks[0] += duration

        class Operator:
            def wait_ready(self, prompt=None):
                if prompt is None:
                    ready.append(driver.pose.copy())

        arm = CLIArm(cfg, home_pose=home_pose, **options, driver_factory=lambda config: driver,
            operator=Operator(), poll_end=lambda: False, sleep_fn=sleep,
            clock=lambda: ticks[0])
        self.addCleanup(arm.close)
        return arm, driver, arm._home_session, sleeps, ready

    def calibration_file(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        ranges = ((991, 3448), (837, 3189), (889, 3103), (862, 3167), (0, 4095), (1576, 3063))
        records = {name: dict(id=index + 1, drive_mode=0, homing_offset=0,
                             range_min=low, range_max=high)
                   for index, (name, (low, high)) in enumerate(zip(packing.MOTORS, ranges, strict=True))}
        path = Path(directory.name) / 'fixture_arm.json'
        path.write_text(json.dumps(records))
        return path, records

    def test_saved_calibration_matches_official_driver_units(self):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.motors_bus import MotorsBus
        from lerobot.motors.feetech.tables import MODEL_RESOLUTION

        path, records = self.calibration_file()
        arm = configured_so101(camera_configs={}, robot_id=path.stem, calibration_dir=str(path.parent))
        self.addCleanup(arm.close)
        bus = SimpleNamespace(
            calibration={name: MotorCalibration(**record) for name, record in records.items()},
            motors={name: Motor(index + 1, 'sts3215', MotorNormMode.DEGREES if index < 5
                                else MotorNormMode.RANGE_0_100)
                    for index, name in enumerate(packing.MOTORS)},
            apply_drive_mode=False, model_resolution_table=MODEL_RESOLUTION,
            _id_to_name=lambda motor_id: packing.MOTORS[motor_id - 1],
            _id_to_model=lambda motor_id: 'sts3215',
        )
        for key, bounds in [('range_min', arm._cfg.low), ('range_max', arm._cfg.high)]:
            normalized = MotorsBus._normalize(bus, {index + 1: records[name][key]
                                                   for index, name in enumerate(packing.MOTORS)})
            np.testing.assert_allclose(bounds, list(normalized.values()))
        self.assertIsNone(arm._driver)
        self.assertIsNone(arm._cfg.max_relative_target)
        self.assertIsNone(arm.info.action_space.semantics.max_step)
        self.assertIn('fixture_arm.json', arm.info.docs)
        self.assertIn('one moving jaw and one fixed jaw', arm.info.docs)

    def test_calibration_ranges_reach_agent_tools(self):
        from inspect_robots import Observation
        from inspect_robots_agent._llm import ToolCall
        from inspect_robots_agent._tools import build_toolset

        path, _ = self.calibration_file()
        arm = configured_so101(camera_configs={}, robot_id=path.stem, calibration_dir=str(path.parent))
        self.addCleanup(arm.close)
        tools = build_toolset(arm.info.action_space, arm.info.observation_space, arm.info.control_hz,
                              max_speed_frac=0.05)
        description = tools.schemas()[0]['function']['description']
        self.assertIn('shoulder_lift: [-103.4, 103.4]', description)
        self.assertIn('elbow_flex: [-97.32, 97.32]', description)
        self.assertIn('gripper: [0, 100]', description)
        observation = Observation(state={'joint_pos': np.array([0, -89, 97, 2, 0, 50], dtype=float)})
        for targets in ({'elbow_flex': 110}, {'shoulder_lift': -115}):
            result = tools.execute(ToolCall(id='offline', name='move_joints',
                arguments=json.dumps({'targets': targets, 'note': 'Replay an unreachable logged target.'})),
                observation)
            self.assertIn('outside', result.error)
            self.assertIsNone(result.chunk)
        result = tools.execute(ToolCall(id='offline', name='move_joints', arguments=json.dumps({
            'targets': {'elbow_flex': 82}, 'note': 'Replay the valid logged elbow extension.'})), observation)
        self.assertIsNone(result.error)
        self.assertEqual(result.chunk.actions[-1].data[2], 82)

    def test_exported_calibration_directory_and_robot_id_select_file(self):
        path, _ = self.calibration_file()
        with patch.dict(os.environ, {'SO101_CALIBRATION_DIR': str(path.parent),
                                     'SO101_ROBOT_ID': path.stem}):
            cfg, _ = rig_config(camera_configs={})
        self.assertAlmostEqual(cfg.joint_high[2], 97.31868131868131)

    def test_normalized_mode_keeps_native_ranges(self):
        path, _ = self.calibration_file()
        cfg, _ = rig_config(camera_configs={}, robot_id=path.stem,
                            calibration_dir=str(path.parent), use_degrees=False)
        self.assertEqual(cfg.joint_low, (-100,) * 5 + (0,))
        self.assertEqual(cfg.joint_high, (100,) * 6)

    def test_explicit_bounds_can_narrow_saved_calibration(self):
        path, _ = self.calibration_file()
        cfg, _ = rig_config(camera_configs={}, robot_id=path.stem, calibration_dir=str(path.parent),
                            joint_low=[-90] * 5 + [0], joint_high=[90] * 5 + [100])
        self.assertEqual(cfg.joint_low, (-90,) * 5 + (0,))
        self.assertEqual(cfg.joint_high, (90,) * 5 + (100,))
        cfg, _ = rig_config(camera_configs={}, robot_id=path.stem, calibration_dir=str(path.parent),
                            joint_low=[-180] * 5 + [0], joint_high=[180] * 5 + [100])
        self.assertAlmostEqual(cfg.joint_high[2], 97.31868131868131)

    def test_invalid_existing_calibration_fails_before_hardware(self):
        path, records = self.calibration_file()
        bad_motor = dict(records['elbow_flex'], range_max=records['elbow_flex']['range_min'])
        for contents in ('bad json', '[]', json.dumps({}),
                         json.dumps(dict(records, elbow_flex=bad_motor))):
            with self.subTest(contents=contents):
                path.write_text(contents)
                with self.assertRaises(ConfigError):
                    configured_so101(camera_configs={}, robot_id=path.stem, calibration_dir=str(path.parent))

    def test_home_must_fit_saved_calibration(self):
        path, _ = self.calibration_file()
        with self.assertRaises(ConfigError):
            rig_config(camera_configs={}, robot_id=path.stem, calibration_dir=str(path.parent),
                       home_pose=[0, 0, 110, 0, 0, 50])

    def test_two_camera_configs_and_upstream_motor_methods(self):
        arm = configured_so101()
        self.assertIsInstance(arm, SOArmEmbodiment)
        self.assertIs(type(arm)._home, SOArmEmbodiment._home)
        self.assertIs(type(arm)._send, SOArmEmbodiment._send)
        self.assertEqual(arm.info.observation_space.camera_names, {'front', 'head'})
        self.assertEqual(arm._cfg.camera_configs['head'].serial_number_or_name, '262522073381')
        self.assertFalse(arm._cfg.camera_configs['head'].use_depth)
        self.assertIsNone(arm._driver)
        arm.close()

    def test_default_eval_limits_match_upstream(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        arm = configured_so101(camera_configs={}, robot_id='missing', calibration_dir=directory.name)
        self.addCleanup(arm.close)
        self.assertIsNone(arm._cfg.max_relative_target)
        self.assertIsNone(arm.info.action_space.semantics.max_step)
        np.testing.assert_array_equal(arm.info.action_space.low, SOArmConfig().low)
        np.testing.assert_array_equal(arm.info.action_space.high, SOArmConfig().high)

    def test_eval_does_not_clip_goal_to_measured_position(self):
        arm, driver, session, _, _ = self.rig(pose=(0, 0, 97, 0, 0, 50))
        with session:
            arm.reset(Scene('test', 'test'))
            driver.stuck_joint = 2
            result = arm.step(Action(np.array([0, 0, 82, 0, 0, 50], dtype=float)))
            self.assertEqual(driver.commands[-1][2], 82)
            self.assertIsNone(driver.command_limits[-1])
            self.assertFalse(result.info['settled'])
            self.assertEqual(result.info['settle_residual'], 15)

    def test_home_limit_is_restored_before_eval(self):
        arm, driver, session, _, _ = self.rig(home_pose=(0, 0, 0, 0, 0, 50),
                                              pose=(3, 0, 0, 0, 0, 50))
        config = driver.config
        arm.reset(Scene('first', 'test'))
        self.assertEqual(driver.command_limits, [1, 1, 1])
        self.assertIs(driver.config, config)
        self.assertIsNone(arm._cfg.max_relative_target)
        arm.step(Action(np.array([8, 0, 0, 0, 0, 50], dtype=float)))
        self.assertEqual(driver.commands[-1][0], 8)
        self.assertIsNone(driver.command_limits[-1])
        arm.reset(Scene('second', 'test'))
        self.assertEqual(driver.command_limits[-8:], [1] * 8)
        self.assertIs(driver.config, config)
        self.assertIsNone(driver.config.max_relative_target)

    def test_failed_home_restores_config_and_disconnects(self):
        arm, driver, session, _, _ = self.rig(home_pose=(0, 0, 0, 0, 0, 50),
                                              pose=(3, 0, 0, 0, 0, 50))
        config = driver.config
        driver.fail_send = True
        with self.assertRaises(OSError), session:
            arm.reset(Scene('test', 'test'))
        self.assertIs(driver.config, config)
        self.assertIsNone(arm._cfg.max_relative_target)
        self.assertTrue(driver.disconnected)

    def test_home_limit_validation(self):
        for value in (None, True, 0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                rig_config(camera_configs={}, home_max_relative_target=value)

    def test_stale_joint_environment_is_ignored(self):
        with patch.dict(os.environ, {'SO101_JOINT_LOW': 'bad', 'SO101_JOINT_HIGH': ''}):
            cfg, _ = rig_config(camera_configs={})
        self.assertEqual(cfg.joint_low, SOArmConfig().joint_low)

    def test_explicit_home_and_limits_validation(self):
        cfg, home = rig_config(home_pose='[0,0,0,0,0,50]', joint_low='[-90,-90,-90,-90,-90,0]')
        self.assertEqual(home['home_pose'], (0, 0, 0, 0, 0, 50))
        self.assertIsNone(cfg.home_pose)
        self.assertIsNone(cfg.max_relative_target)
        for value in ('', '[0,0]', '[0,0,0,0,0,NaN]', '[0,0,0,0,0,101]'):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                rig_config(home_pose=value)

    def test_capture_home_and_repeat_reset_ramp(self):
        arm, driver, session, sleeps, ready = self.rig()
        with session:
            obs = arm.reset(Scene('first', 'test'))
            self.assertEqual(set(obs.images), {'front', 'head'})
            self.assertEqual(session.home_pose, (0, 0, 0, 0, 0, 50))
            driver.pose[0] = 4
            driver.commands.clear()
            arm.reset(Scene('second', 'test'))
            np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [3, 2, 1, 0])
            np.testing.assert_allclose(ready[-1], session.home_pose)
        self.assertTrue(all(duration >= 0.1 - 1e-10 for duration in sleeps))
        self.assertTrue(driver.disconnected)

    def test_reconnect_captures_new_starting_pose(self):
        arm, driver, session, _, _ = self.rig(return_home=False)
        arm.reset(Scene('first', 'test'))
        np.testing.assert_allclose(session.home_pose, (0, 0, 0, 0, 0, 50))
        arm.close()
        driver.pose = np.array([5, -10, 15, 20, -5, 25], dtype=float)
        driver.commands.clear()
        arm.reset(Scene('reconnected', 'test'))
        np.testing.assert_allclose(session.home_pose, driver.pose)
        np.testing.assert_allclose(session.home_pose, (5, -10, 15, 20, -5, 25))
        np.testing.assert_allclose(driver.commands[-1], session.home_pose)

    def test_explicit_home_is_preserved_after_reconnect(self):
        fixed_home = (0, 0, 0, 0, 0, 50)
        arm, driver, session, _, _ = self.rig(home_pose=fixed_home, return_home=False)
        arm.reset(Scene('first', 'test'))
        arm.close()
        driver.pose[0] = 3
        driver.commands.clear()
        arm.reset(Scene('reconnected', 'test'))
        self.assertEqual(session.home_pose, fixed_home)
        np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [2, 1, 0])

    def test_normal_completion_returns_home_then_disconnects(self):
        arm, driver, session, _, _ = self.rig()
        with session:
            arm.reset(Scene('test', 'test'))
            arm.step(Action(np.array([1, 0, 0, 0, 0, 50], dtype=float)))
            driver.pose[0] = 3
            driver.commands.clear()
            session.completed = True
        np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [2, 1, 0])
        self.assertTrue(driver.disconnected)
        count = len(driver.commands)
        arm.close()
        self.assertEqual(len(driver.commands), count)

    def test_saved_home_used_before_readiness(self):
        arm, driver, session, _, ready = self.rig(home_pose=(0, 0, 0, 0, 0, 50), pose=(3, 0, 0, 0, 0, 50))
        with session:
            arm.reset(Scene('test', 'test'))
            np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [2, 1, 0])
            np.testing.assert_allclose(ready[0], (0, 0, 0, 0, 0, 50))

    def test_fault_skips_shutdown_motion(self):
        arm, driver, session, _, _ = self.rig()
        with self.assertRaises(OSError), session:
            arm.reset(Scene('test', 'test'))
            driver.fail_send = True
            arm.step(Action(np.zeros(6)))
        self.assertTrue(driver.disconnected)
        self.assertEqual(len(driver.commands), 1)

    def test_interrupt_returns_home_before_disconnect(self):
        arm, driver, session, _, _ = self.rig()
        with self.assertRaises(KeyboardInterrupt), session:
            arm.reset(Scene('test', 'test'))
            driver.pose[0] = 3
            driver.commands.clear()
            raise KeyboardInterrupt()
        np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [2, 1, 0])
        self.assertTrue(driver.disconnected)
        self.assertIsNone(arm._driver)

    def test_error_log_skips_shutdown_motion(self):
        arm, driver, session, _, _ = self.rig()
        with session:
            arm.reset(Scene('test', 'test'))
            driver.pose[0] = 4
            session.completed = False
        self.assertEqual(len(driver.commands), 1)
        self.assertTrue(driver.disconnected)

    def test_disconnect_after_failed_shutdown_home(self):
        arm, driver, session, _, _ = self.rig()
        with self.assertRaises(OSError), session:
            arm.reset(Scene('test', 'test'))
            driver.fail_send = True
            session.completed = True
        self.assertTrue(driver.disconnected)
        self.assertIsNone(arm._driver)

    def test_invalid_start_fails_without_motion(self):
        _, driver, session, _, _ = self.rig(pose=(float('nan'), 0, 0, 0, 0, 0))
        with self.assertRaises(ValueError):
            session.__enter__()
        self.assertEqual(driver.commands, [])
        self.assertTrue(driver.disconnected)

    def test_disable_shutdown_home(self):
        arm, driver, session, _, _ = self.rig(return_home=False)
        with session:
            arm.reset(Scene('test', 'test'))
            driver.pose[0] = 3
            session.completed = True
        self.assertEqual(len(driver.commands), 1)

    def test_close_without_eval_does_not_connect(self):
        arm, driver, _, _, _ = self.rig()
        arm.close()
        self.assertIsNone(arm._driver)
        self.assertFalse(driver.disconnected)

    def test_both_camera_images_reach_agent_content(self):
        from inspect_robots_agent.policy import _observation_content

        arm, _, session, _, _ = self.rig()
        with session:
            observation = arm.reset(Scene('test', 'test'))
            content = _observation_content(observation)
        self.assertEqual(sum(part['type'] == 'image_url' for part in content), 2)
        labels = [part['text'] for part in content if part['type'] == 'text']
        self.assertIn("camera 'front':", labels)
        self.assertIn("camera 'head':", labels)

    def test_cli_adapter_reset_and_shutdown_use_upstream_ramp(self):
        base, driver, _, sleeps, _ = self.rig()
        arm = CLIArm(base._cfg, driver_factory=lambda cfg: driver,
                     operator=base._operator, poll_end=lambda: False,
                     sleep_fn=base._sleep, clock=base._clock)
        self.addCleanup(arm.close)
        arm.reset(Scene('test', 'test'))
        driver.pose[0] = 3
        driver.commands.clear()
        arm.close()
        np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [2, 1, 0])
        self.assertTrue(driver.disconnected)
        self.assertTrue(sleeps)

    def test_cli_interrupt_returns_home_after_cli_catches_it(self):
        arm, driver, _, _, _ = self.rig()
        previous = signal.getsignal(signal.SIGINT)
        arm.reset(Scene('test', 'test'))
        driver.pose[0] = 3
        driver.commands.clear()
        with self.assertRaises(KeyboardInterrupt):
            signal.default_int_handler(signal.SIGINT, None)
        # Like the CLI: the exception has already been handled before close().
        arm.close()
        np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [2, 1, 0])
        self.assertTrue(driver.disconnected)
        self.assertIs(signal.getsignal(signal.SIGINT), previous)
        count = len(driver.commands)
        arm.close()
        self.assertEqual(len(driver.commands), count)

    def test_interrupt_during_step_returns_home(self):
        arm, driver, _, _, _ = self.rig()
        arm.reset(Scene('test', 'test'))
        driver.pose[0] = 3
        driver.commands.clear()
        with patch.object(driver, 'send_action', side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                arm.step(Action(np.array([4, 0, 0, 0, 0, 50], dtype=float)))
        self.assertFalse(arm._motion_failed)
        arm.close()
        np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [2, 1, 0])
        self.assertTrue(driver.disconnected)

    def test_arm_context_manager_returns_home_on_interrupt(self):
        arm, driver, _, _, _ = self.rig()
        with self.assertRaises(KeyboardInterrupt), arm:
            arm.reset(Scene('test', 'test'))
            driver.pose[0] = 3
            driver.commands.clear()
            raise KeyboardInterrupt()
        np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [2, 1, 0])
        self.assertTrue(driver.disconnected)

    def test_second_interrupt_aborts_shutdown_ramp_and_disconnects(self):
        arm, driver, _, _, _ = self.rig()
        arm.reset(Scene('test', 'test'))
        driver.pose[0] = 4
        driver.commands.clear()
        send = driver.send_action

        def send_then_interrupt(action):
            if driver.commands:
                signal.default_int_handler(signal.SIGINT, None)
            return send(action)

        with patch.object(driver, 'send_action', side_effect=send_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                arm.close()
        np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [3])
        self.assertTrue(driver.disconnected)
        self.assertIsNone(arm._driver)
        self.assertIsNone(driver.config.max_relative_target)
        arm.close()
        self.assertEqual(len(driver.commands), 1)

    def test_return_home_false_skips_interrupted_shutdown(self):
        arm, driver, session, _, _ = self.rig(return_home=False)
        with self.assertRaises(KeyboardInterrupt), session:
            arm.reset(Scene('test', 'test'))
            driver.pose[0] = 3
            driver.commands.clear()
            raise KeyboardInterrupt()
        self.assertEqual(driver.commands, [])
        self.assertTrue(driver.disconnected)


    def test_fixed_home_gate_precedes_ramp_and_scene_ready(self):
        from inspect_robots_so101.operator import OperatorIO

        arm, driver, _, _, _ = self.rig(home_pose=(0, 0, 0, 0, 0, 50),
                                        capture_home=False, pose=(3, 0, 0, 0, 0, 50))
        events = []
        def read(prompt):
            events.append(('prompt', prompt, driver.pose.copy(), len(driver.commands)))
            return ''
        arm._operator = OperatorIO(input_fn=read, output_fn=lambda line: events.append(('status', line)))
        observed = arm.reset(Scene('test', 'test'))
        self.assertIn('Arm will move to the home pose', events[0][1])
        self.assertEqual(events[0][3], 0)
        self.assertIn('ramping SO-101 to home position', events[1][1])
        self.assertIn('Position the scene', events[2][1])
        self.assertEqual(events[2][3], 3)
        np.testing.assert_allclose(events[2][2], (0, 0, 0, 0, 0, 50))
        np.testing.assert_allclose(observed.state['joint_pos'], events[2][2])
        np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [2, 1, 0])
        self.assertEqual(driver.command_limits, [1, 1, 1])
        self.assertIsNone(driver.config.max_relative_target)
        arm.close()
        self.assertIn('ramping SO-101 back to home position', events[-1][1])

    def test_home_gate_once_per_connection_and_ramp_each_trial(self):
        from inspect_robots_so101.operator import OperatorIO

        arm, driver, _, _, _ = self.rig(home_pose=(0, 0, 0, 0, 0, 50), capture_home=False)
        prompts = []
        lines = []
        arm._operator = OperatorIO(input_fn=lambda prompt: prompts.append(prompt) or '',
                                   output_fn=lines.append)
        arm.reset(Scene('first', 'test'))
        driver.pose[0] = 3
        driver.commands.clear()
        arm.reset(Scene('second', 'test'))
        self.assertEqual(sum('Arm will move' in prompt for prompt in prompts), 1)
        self.assertEqual(sum('Position the scene' in prompt for prompt in prompts), 2)
        np.testing.assert_allclose([cmd[0] for cmd in driver.commands], [2, 1, 0])
        self.assertEqual(sum('to home position' in line for line in lines), 2)
        arm.close()
        arm.reset(Scene('reconnected', 'test'))
        self.assertEqual(sum('Arm will move' in prompt for prompt in prompts), 2)

    def test_cancel_home_gate_sends_no_position_commands(self):
        from inspect_robots_so101.operator import OperatorIO

        arm, driver, _, _, _ = self.rig(home_pose=(0, 0, 0, 0, 0, 50),
                                        capture_home=False, pose=(3, 0, 0, 0, 0, 50))
        arm._operator = OperatorIO(input_fn=Mock(side_effect=KeyboardInterrupt()), output_fn=Mock())
        with self.assertRaises(KeyboardInterrupt):
            arm.reset(Scene('test', 'test'))
        arm.close()
        self.assertEqual(driver.commands, [])
        self.assertTrue(driver.disconnected)
        self.assertFalse(arm._home_gate_confirmed)
        self.assertIsNone(driver.config.max_relative_target)
        # A later connection gets a fresh gate and can complete normally.
        arm._operator.input_fn = Mock(return_value='')
        arm.reset(Scene('retry', 'test'))
        self.assertFalse(arm._motion_failed)
        self.assertEqual(arm._operator.input_fn.call_count, 2)

    def test_g05_explicit_home_is_valid_in_native_calibration_units(self):
        path, _ = self.calibration_file()
        home = [-1.2449, -10.5453, 3.1424, 77.0071, -0.8653, 5.7005]
        arm = configured_so101(camera_configs={}, robot_id=path.stem, calibration_dir=str(path.parent),
                               home_pose=home, capture_home=False)
        self.addCleanup(arm.close)
        np.testing.assert_allclose(arm._home_session.home_pose, home)
        self.assertFalse(arm._home_session.capture_home)
        self.assertIsNone(arm._cfg.max_relative_target)
        self.assertIn('Configured home pose', arm.info.docs)
        self.assertIsNone(arm._driver)


class InitializationTests(unittest.TestCase):
    def robot(self):
        robot = Mock()
        robot.config.disable_torque_on_disconnect = False
        robot.is_calibrated = True
        robot.bus.is_connected = True
        front = Mock(is_connected=True, thread=object())
        head = Mock(is_connected=False, thread=None)
        robot.cameras = {'front': front, 'head': head}
        return robot, front, head

    def test_default_factory_keeps_official_config_and_calibration_check(self):
        cfg, _ = rig_config(camera_configs={}, robot_id='fixture', calibration_dir='/tmp')
        robot, front, head = self.robot()
        with patch('lerobot.robots.so_follower.SOFollower', return_value=robot) as factory:
            self.assertIs(_connected_driver(cfg), robot)
        robot.connect.assert_called_once_with(calibrate=False)
        supplied = factory.call_args.args[0]
        self.assertEqual(supplied.id, cfg.robot_id)
        self.assertEqual(supplied.calibration_dir, Path(cfg.calibration_dir))
        self.assertEqual(supplied.port, cfg.port)
        self.assertEqual(supplied.cameras, cfg.camera_configs)
        self.assertEqual(supplied.use_degrees, cfg.use_degrees)
        self.assertIsNone(supplied.max_relative_target)
        self.assertEqual(supplied.disable_torque_on_disconnect, cfg.disable_torque_on_disconnect)
        front.disconnect.assert_not_called()
        head.disconnect.assert_not_called()
        robot.bus.disconnect.assert_not_called()
        arm = CLIArm(cfg)
        self.addCleanup(arm.close)
        self.assertIs(arm._driver_factory, _connected_driver)
        self.assertIsNone(arm._driver)

    def test_second_camera_failure_closes_front_and_bus_and_reports_sdk_cause(self):
        robot, front, head = self.robot()
        error = ConnectionError('Failed to open RealSenseCamera')
        error.__cause__ = RuntimeError('Device or resource busy')
        robot.connect.side_effect = error
        with patch('lerobot.robots.so_follower.SOFollower', return_value=robot):
            with self.assertRaisesRegex(ConnectionError, 'Device or resource busy') as caught:
                _connected_driver(SOArmConfig())
        self.assertIs(caught.exception.__cause__, error)
        front.disconnect.assert_called_once_with()
        head.disconnect.assert_not_called()
        robot.bus.disconnect.assert_called_once_with(False)
        robot.send_action.assert_not_called()

    def test_mismatched_calibration_also_releases_components(self):
        robot, front, head = self.robot()
        robot.is_calibrated = False
        head.is_connected = True
        with patch('lerobot.robots.so_follower.SOFollower', return_value=robot):
            with self.assertRaisesRegex(RuntimeError, 'not calibrated'):
                _connected_driver(SOArmConfig())
        front.disconnect.assert_called_once_with()
        head.disconnect.assert_called_once_with()
        robot.bus.disconnect.assert_called_once_with(False)
        robot.send_action.assert_not_called()

    def test_motor_connection_failure_does_not_disconnect_unopened_devices(self):
        robot, front, head = self.robot()
        robot.bus.is_connected = False
        front.is_connected = False
        front.thread = None
        error = OSError('Missing motor IDs')
        robot.connect.side_effect = error
        with patch('lerobot.robots.so_follower.SOFollower', return_value=robot):
            with self.assertRaises(OSError) as caught:
                _connected_driver(SOArmConfig())
        self.assertIs(caught.exception, error)
        front.disconnect.assert_not_called()
        head.disconnect.assert_not_called()
        robot.bus.disconnect.assert_not_called()

    def test_cancelled_initialization_closes_camera_thread_without_motion(self):
        robot, front, head = self.robot()
        front.is_connected = False
        robot.connect.side_effect = KeyboardInterrupt()
        with patch('lerobot.robots.so_follower.SOFollower', return_value=robot):
            with self.assertRaises(KeyboardInterrupt):
                _connected_driver(SOArmConfig())
        front.disconnect.assert_called_once_with()
        robot.bus.disconnect.assert_called_once_with(False)
        robot.send_action.assert_not_called()

    def test_cleanup_failure_keeps_original_error_and_closes_other_components(self):
        robot, front, head = self.robot()
        head.is_connected = True
        front.disconnect.side_effect = RuntimeError('front cleanup failed')
        error = TimeoutError('head warmup failed')
        robot.connect.side_effect = error
        with patch('lerobot.robots.so_follower.SOFollower', return_value=robot):
            with self.assertLogs('_so101_motion', level='WARNING'):
                with self.assertRaises(TimeoutError) as caught:
                    _connected_driver(SOArmConfig())
        self.assertIs(caught.exception, error)
        head.disconnect.assert_called_once_with()
        robot.bus.disconnect.assert_called_once_with(False)


if __name__ == '__main__':
    unittest.main()
