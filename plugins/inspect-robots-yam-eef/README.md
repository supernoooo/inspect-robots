# inspect-robots-yam-eef

This plugin registers `yam_eef`, a separately selectable EEF control mode for
the I2RT YAM bimanual arms. It uses the tested Cartesian implementation in
`inspect-robots-yam`: the agent emits a 14-value absolute EEF target; YAM
uses measured joint state and its previous command to solve IK, then sends a
bounded joint target to the two motor controllers. The robot remains responsible
for cameras, homing, operator interaction, and joint limits.

Install into the environment that already contains `inspect-robots-yam` and
`inspect-robots-agent`:

```bash
uv pip install -e ./plugins/inspect-robots-yam-eef
```

By default, `yam_eef` uses the same joint-space home as YAM's joint-control
mode: six zero joint angles and an open gripper for each arm. This overrides
YAM's provisional EEF-specific home, which has different joint angles. You
can supply a rig-validated 14-value `home_pose` or saved YAM `start_pose`
to override the default. Both poses use `[left_j0..j5, left_gripper, right_j0..j5,
right_gripper]` (radians, with grippers normalized 0 closed to 1 open). The
selected pose must also lie within the configured EEF workspace after FK.

Run the generic agent with the new embodiment name:

```bash
inspect-robots run \
  --instruction "Pick up the block and place it back on the table." \
  --policy agent --embodiment yam_eef \
  -P model=openai/gpt-6-astra -P wire=responses \
  -P images=always -P max_speed_frac=0.25 \
  --max-steps 2000 --scorer operator
```

Your existing `[embodiment.args]` camera, CAN, and other applicable YAM
settings still apply. The default `rest_pose` is also YAM's zero-joint,
open-gripper pose. Set `rest_pose=none` to park at the joint pose captured
before reset, or supply an explicit pose after validating its ramp and
gravity-stable endpoint. Remove `control_interface = joints` if it is
present: `yam_eef` fixes that setting to `eef_pos`. The explicit CLI option
`-E control_interface=eef_pos` is also accepted. Inspect the selected contract
without moving the robot:

```bash
inspect-robots list embodiments
inspect-robots doctor --embodiment yam_eef
```

The agent receives `joint_pos` and `eef_state` observations and uses `move_to`
with `left_x`, `left_y`, `left_z`, `left_yaw`, `left_pitch`, `left_roll`,
`left_gripper`, and the matching `right_*` fields. Each arm's x/y/z values are
metres in its **own base frame**, not a shared world frame. Orientation values
are radians relative to the pose captured at reset. Gripper values range from
0 (closed) to 1 (open). The default pitch and roll axes are pinned at zero;
`-E eef_orientation=true` opens them, subject to workspace configuration.
The system prompt also describes the YAM grasp-site tool axes and tells the
agent to choose an approach from live images and verify it against measured
`eef_state`. It does not supply an object pose or a preplanned grasp. With the
default zero-joint home, the tool approaches horizontally. When
`eef_orientation=true` opens the otherwise default bounds, pitch is limited
to +/-0.6 rad, or about 34 degrees of tilt. A nearly vertical
top-down approach needs a rig-validated wider pitch workspace and reachable
IK at a collision-free height. Do not infer that from prompt text alone.

The action space's workspace bounds and the standard CLI clamp and step
limits still apply. The YAM IK layer limits per-step joint motion and holds
the previous arm command for nonfinite solutions. YAM's EEF mode currently
does **not** run its joint-space arm/table or arm/arm collision approver; its
IK may also return a limited iterate for a target it cannot reach exactly.
Use small moves, reobserve after each chunk, and validate the workspace and
table height on the actual rig with an operator at the e-stop. See the
`Cartesian EEF mode` section in the `inspect-robots-yam` README for the full
coordinate and safety contract.
