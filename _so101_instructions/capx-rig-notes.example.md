# SO-101 rig notes (measured on this setup, 2026-11-06; revised after run006 and run007)

## Cameras
- 'front' (RealSense, has depth) sits BEHIND the robot and looks forward over it.
  With the arm folded it sees the whole table, so use it at the start to find
  objects (e.g. segment('blue block') scored 0.97) and their depth.
- Once the arm is raised it blocks the view and the jaws face away from 'front':
  segment('robot gripper') scores ~0 there and objects behind the arm read the
  arm's depth instead (0.32 m instead of 0.63 m). Do not use 'front' to locate
  the gripper.
- 'wrist' (no depth) is rigidly mounted on the gripper. Only ONE jaw is fixed:
  it is the dark jaw at the bottom left of the wrist image (from about u 180 to
  305, v 345 to 480), tip near (300, 350), inner face pointing to +u. The
  moving jaw swings in from the RIGHT; when the gripper is open it is out of
  the image, when closed it appears at about u 315-455, v 355-480.
  segment('blue block', camera='wrist') works well (0.95-0.98); the jaws need
  no segmentation.
- GRASP ALIGNMENT (wrist image, gripper open, just before closing). Two axes
  matter and the wrist camera has NO depth:
  * Across the jaws (u): having the fixed jaw on the block's LEFT side is
    correct, because the moving jaw closes from the right. Keep the block's
    left edge just right of the fixed jaw's inner face (u ~310-320).
  * Along the fingers (v): the block must be level with the jaw TIPS, not
    near the bottom of the image. In run007, three grasps with the block's
    centroid at v ~430-450 (bottom edge cut off by the image border) all
    closed empty (gripper read 0.8-3.5): the jaws met in front of the block.
    Before closing, the whole block must be visible with its centroid well
    above v 420, overlapping the tip region (tips at v ~350). If the block
    drifts toward the bottom while descending, stop, rise 2 cm and correct
    first.
  * The gripper reads ~0-4 when it closes on nothing; a held 4 cm block keeps
    it clearly higher. Check this after close_gripper() before lifting.

## Geometry (robot-base frame, metres; x forward, y left, z up)
- The table surface is roughly z = -0.01 to 0.00 (the folded rest pose puts
  the tool at z = -0.007). The blue block is about 4 cm wide, so grasp its
  centre around z = 0.015-0.02 and keep z >= 0.08 while still aligning.
- Comfortable reach: x 0.12-0.30, |y| < 0.15. Straight-down orientation is
  close to quaternion_wxyz [0, 0, 1, 0]; IK may land ~20 deg off it, which is
  fine for a top-down grasp.
- Under gravity the arm settles ~5 mm lower than commanded when extended.

## Suggested grasp loop
1. With the arm still folded, segment the object on 'front' to confirm it is
   there and roughly where (left/right of centre, near/far by depth).
2. Raise the gripper to ~10 cm above the table near the middle (x ~0.20, y 0)
   with open_gripper().
3. Image-to-motion mapping measured at z ~ 0.095 m, straight-down tool
   (wrist pixels per 1 cm of base motion): +x moves the object about -24.5 u
   and -0.8 v; +y moves it about -4.5 u and +26.5 v. The scale grows as the
   gripper gets lower, so re-measure with one 2 cm probe if steps overshoot.
4. Step (<= 3 cm) until the object reaches the grasp alignment above, then
   descend in 2-3 cm steps to z ~ 0.015-0.02, re-checking alignment each turn
   (the object moves in the image as you descend).
5. close_gripper(), lift 5 cm, and confirm in 'wrist' that the object moved up
   with the jaws before replying FINISH.
