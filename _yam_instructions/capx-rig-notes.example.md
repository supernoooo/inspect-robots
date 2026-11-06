# YAM rig notes (TEMPLATE: measure on your rig and replace every <...>)

These notes are appended to the model's system prompt through
`prior_learnings`. Everything below steers every model equally, so a wrong
number becomes a systematic failure. Measure, then write; do not guess.

## Cameras
- 'top_cam' (RealSense, has depth) is mounted <where relative to the driven
  arm's base: in front / behind / above, looking in which direction>. With
  both arms at the folded home it sees <the whole table / part of it>, so use
  it at the start to find objects and read their depth. Check whether the
  raised arm occludes it.
- 'left_cam' / 'right_cam' (RealSense, has depth) are mounted <on the wrist of
  that arm / fixed next to it>. If a wrist camera is rigid with the gripper,
  the jaws stay at a fixed spot in its image: fixed jaw at about (u <..>,
  v <..>), moving jaw closes from the <left/right>. Read these off a stored
  frame (`logs/.../frames/*_left_cam_*.npy`) or the Rerun viewer.
- Grasp alignment in the wrist image (gripper open, just before closing):
  the object must sit between the jaw tips, not near the image border; note
  the pixel band that works (u <..>-<..>, v <..>-<..>) once a grasp succeeds.
- The gripper reads about <..> when it closes on nothing and clearly higher
  on a held <size> object; check this after close_gripper() before lifting.

## Geometry (driven arm's base frame, metres; x forward, y left, z up)
- The table surface is at z = <..> (the folded home puts the grasp point at
  z = <..>; the `collision_table_height` you measured is the same number).
- At zero joints the tool +z (approach axis) points along base +x, so a
  top-down grasp needs the tool rotated to point +z down: quaternion_wxyz
  [0, 0, 1, 0] closes the jaws along base y; yaw that quaternion about base z
  to pick the jaw direction. Keep z >= <..> while aligning above an object.
- Comfortable reach: x <..>-<..>, |y| < <..>. Commands outside the joint
  limits raise in solve_ik.
- The idle arm holds its start pose at <where>; keep targets clear of it.

## Suggested grasp loop
1. With the arm still folded, segment the object on 'top_cam' to confirm it
   is there and roughly where (left/right of centre, near/far by depth).
2. Raise the gripper to about 10 cm above the table near the middle of the
   workspace with open_gripper() and the approach axis pointing down.
3. Measure the image-to-motion mapping with one 2 cm probe along a base axis
   and keep the result in variables.
4. Step (<= 3 cm) until the object reaches the grasp alignment above, then
   descend in 2-3 cm steps, re-checking alignment each turn.
5. close_gripper(), lift 5 cm, and confirm in the wrist camera that the object
   moved up with the jaws before replying FINISH.
