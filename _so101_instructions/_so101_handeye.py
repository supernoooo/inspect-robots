"""Eye-to-hand calibration of the fixed front RealSense against the SO-101 base.

An ArUco marker taped near the gripper tip is observed at several arm poses
while the leader arm teleoperates the follower. Each capture pairs the
marker's 3-D camera-frame point (aligned depth + color intrinsics, the same
model ``capx`` uses at run time) with the follower's measured joints. The
solver then finds the camera-to-base transform together with the marker's
unknown offset in the gripper frame, so the marker never has to be measured.

Usage (from the repository root, with .venv active)::

    python _so101_instructions/_so101_handeye.py --print-marker marker.png
    python _so101_instructions/_so101_handeye.py --config .so101/capx.ini
    python _so101_instructions/_so101_handeye.py --config .so101/capx.ini --verify

The live camera view (marker outline, and in ``--verify`` the predicted
marker position) appears in a Rerun window; commands are typed in the
terminal: Enter captures, ``u`` undoes, ``q`` solves and saves, ``x`` aborts.
Captures are written to the samples file after every capture, so
``--solve-only`` can re-solve them (optionally with ``--drop``) offline.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

MIN_SAMPLES = 8
MIN_DEPTH_FRACTION = 0.3
SETTLE_DEG = 0.5
TRACKING_DEG = 3.0
GOOD_RMS_M = 0.005
WARN_RMS_M = 0.010
WEAK_ROTATION = 0.3
_REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _REPO / ".so101/front_camera_to_base.json"
DEFAULT_SAMPLES = _REPO / ".so101/handeye_samples.json"
ARUCO_DICT = "DICT_4X4_50"
COMMANDS = (
    "Terminal commands: Enter = capture this pose, u = undo last, "
    "q = solve and save, x = abort. The live view is in the Rerun window."
)
COMMANDS_VERIFY = "Terminal commands: Enter = print current error, q = quit, x = abort."


# --- geometry (pure, unit tested) -----------------------------------------------------


def deproject(pixel: tuple[float, float], depth_m: float, intrinsics: np.ndarray) -> np.ndarray:
    """Return the camera-frame 3-D point (m) of a pixel at a metric depth (pinhole, no skew)."""
    u, v = pixel
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    return np.array([(u - cx) / fx * depth_m, (v - cy) / fy * depth_m, depth_m])


def project(point: np.ndarray, intrinsics: np.ndarray) -> tuple[float, float]:
    """Return the pixel of a camera-frame point in front of the camera."""
    x, y, z = point
    return (
        float(intrinsics[0, 0] * x / z + intrinsics[0, 2]),
        float(intrinsics[1, 1] * y / z + intrinsics[1, 2]),
    )


def marker_point(
    corners: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the marker-center camera point and the fraction of valid depth used.

    Depth is the median over the marker's central region (corners shrunk
    halfway toward the center), so edge pixels and holes do not bias it.
    Raises ``ValueError`` when too little of that region has depth.
    """
    import cv2

    quad = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center = quad.mean(axis=0)
    inner = center + 0.5 * (quad - center)
    mask = np.zeros(depth_m.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(inner).astype(np.int32), 1)
    values = depth_m[mask.astype(bool)]
    if values.size == 0:
        raise ValueError("marker is too small in the image")
    valid = values[values > 0]
    fraction = float(valid.size / values.size)
    if fraction < MIN_DEPTH_FRACTION:
        raise ValueError(f"only {fraction:.0%} of the marker has depth; tilt it toward the camera")
    return deproject((float(center[0]), float(center[1])), float(np.median(valid)), intrinsics), (
        fraction
    )


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the 4x4 rigid transform minimizing ||T @ source - target|| (no scale)."""
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - src_mean).T @ (target - dst_mean)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = dst_mean - rotation @ src_mean
    return transform


@dataclass(frozen=True)
class Solution:
    """Camera-to-base transform, marker offset, and per-sample residuals (m)."""

    camera_to_base: np.ndarray
    marker_offset: np.ndarray
    residuals: np.ndarray
    rotation_spread: float

    @property
    def rms(self) -> float:
        """Root-mean-square residual in meters."""
        return float(np.sqrt(np.mean(self.residuals**2)))


def solve(camera_points: np.ndarray, tool_poses: np.ndarray, iterations: int = 200) -> Solution:
    """Fit ``base = T @ cam`` where each marker sits at ``tool_pose @ offset``.

    Alternates a Kabsch fit of T (offset fixed) with a linear least-squares
    fit of the gripper-frame offset (T fixed). The offset is only observable
    when the tool orientation varies across samples; ``rotation_spread`` (the
    smallest singular value of the centered stacked rotations) reports how
    well, and a weak spread falls back to a small ridge toward zero offset.
    """
    points = np.asarray(camera_points, dtype=np.float64)
    poses = np.asarray(tool_poses, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 4
        or poses.shape
        != (
            len(points),
            4,
            4,
        )
    ):
        raise ValueError("need at least 4 samples with (N, 3) points and (N, 4, 4) poses")
    rotations, translations = poses[:, :3, :3], poses[:, :3, 3]
    stacked = rotations.reshape(-1, 3)
    centered = (rotations - rotations.mean(axis=0)).reshape(-1, 3)
    spread = float(np.linalg.svd(centered, compute_uv=False)[-1])
    ridge = 0.0 if spread >= WEAK_ROTATION else 1e-2 * len(points)
    offset = np.zeros(3)
    transform = np.eye(4)
    for _ in range(iterations):
        markers = np.einsum("nij,j->ni", rotations, offset) + translations
        transform = kabsch(points, markers)
        mapped = points @ transform[:3, :3].T + transform[:3, 3]
        rhs = (mapped - translations).reshape(-1)
        normal = stacked.T @ stacked + ridge * np.eye(3)
        updated = np.linalg.solve(normal, stacked.T @ rhs)
        if np.linalg.norm(updated - offset) < 1e-7:
            offset = updated
            break
        offset = updated
    markers = np.einsum("nij,j->ni", rotations, offset) + translations
    transform = kabsch(points, markers)
    mapped = points @ transform[:3, :3].T + transform[:3, 3]
    residuals = np.linalg.norm(mapped - markers, axis=1)
    return Solution(transform, offset, residuals, spread)


def solve_samples(samples: list[dict[str, Any]], kinematics: Any) -> Solution:
    """Solve recorded samples, computing each tool pose by forward kinematics."""
    points = np.array([sample["camera_point_m"] for sample in samples], dtype=np.float64)
    poses = np.array(
        [kinematics.forward_kinematics(np.asarray(s["joints_deg"][:5])) for s in samples]
    )
    return solve(points, poses)


def result_document(solution: Solution, **metadata: Any) -> dict[str, Any]:
    """Return the JSON document ``load_extrinsics`` reads (``camera_to_base`` key)."""
    return {
        "camera_to_base": solution.camera_to_base.tolist(),
        "rms_m": solution.rms,
        "max_residual_m": float(solution.residuals.max()),
        "residuals_m": solution.residuals.tolist(),
        "marker_offset_m": solution.marker_offset.tolist(),
        "rotation_spread": solution.rotation_spread,
        "n_samples": len(solution.residuals),
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        **metadata,
    }


def report(solution: Solution, dropped: list[int]) -> str:
    """Render a human-readable quality summary with the next action."""
    lines = [
        f"samples used: {len(solution.residuals)}" + (f" (dropped {dropped})" if dropped else "")
    ]
    for index, value in enumerate(solution.residuals):
        flag = "  <-- large" if value > 2 * max(np.median(solution.residuals), 1e-4) else ""
        lines.append(f"  #{index:2d}  {value * 1000:6.1f} mm{flag}")
    rms_mm = solution.rms * 1000
    lines.append(f"RMS {rms_mm:.1f} mm, max {solution.residuals.max() * 1000:.1f} mm")
    lines.append(
        "marker offset in gripper frame (mm): "
        + str(np.round(solution.marker_offset * 1000, 1).tolist())
    )
    position = solution.camera_to_base[:3, 3]
    lines.append(f"camera position in base frame (m): {np.round(position, 3).tolist()}")
    if solution.rotation_spread < WEAK_ROTATION:
        lines.append(
            "WARNING: wrist orientation barely varied, so the marker offset is poorly "
            "determined. Recollect with more wrist_flex/wrist_roll variety."
        )
    if solution.rms <= GOOD_RMS_M:
        lines.append("quality: good")
    elif solution.rms <= WARN_RMS_M:
        lines.append("quality: usable; drop flagged samples or add more spread-out poses")
    else:
        lines.append("quality: poor; check the marker is rigid and recollect")
    return "\n".join(lines)


def write_marker(path: Path, marker_id: int = 0, pixels: int = 600) -> None:
    """Write a printable ArUco marker PNG with a white quiet zone."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, pixels)
    border = pixels // 4
    image = cv2.copyMakeBorder(
        marker, border, border, border, border, cv2.BORDER_CONSTANT, value=255
    )
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def detect(rgb: np.ndarray, marker_id: int) -> np.ndarray | None:
    """Return the (4, 2) corner pixels of ``marker_id`` in an RGB image, if visible."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return None
    for quad, found in zip(corners, ids.reshape(-1), strict=True):
        if int(found) == marker_id:
            return np.asarray(quad, dtype=np.float64).reshape(4, 2)
    return None


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Read saved captures (empty when the file does not exist)."""
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["samples"])


def save_samples(path: Path, samples: list[dict[str, Any]], **metadata: Any) -> None:
    """Atomically write captures so a crash never loses earlier ones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({**metadata, "samples": samples}, indent=1), encoding="utf-8")
    os.replace(temporary, path)


# --- rig and hardware loop ------------------------------------------------------------


def rig_settings(config_path: str) -> dict[str, Any]:  # pragma: no cover - reads local config
    """Resolve follower, leader, camera and URDF settings from an inspect-robots INI."""
    from _run_so101 import _DEFAULT_URDF, rig_config

    from inspect_robots.defaults import load_defaults

    env = {**os.environ, "INSPECT_ROBOTS_CONFIG": str(Path(config_path).resolve())}
    defaults = load_defaults(env)
    if defaults.embodiment_args_owner not in {"so101_capx_configured", "so101_configured"}:
        raise SystemExit(f"{config_path} does not configure an SO-101 embodiment")
    args = dict(defaults.embodiment_args)
    urdf = args.pop("urdf_path", None) or _DEFAULT_URDF
    for key in ("camera_extrinsics", "ik_position_tolerance_m", "ik_orientation_weight"):
        args.pop(key, None)
    serial = args.get("head_camera_serial") or os.environ.get("SO101_HEAD_CAMERA_SERIAL")
    if not serial:
        raise SystemExit("head_camera_serial is not configured; set it in the config file")
    serial = str(serial)
    config, home = rig_config(**args)
    if not home.get("leader_port"):
        raise SystemExit("leader_port is not configured; calibration uses leader teleoperation")
    return {"config": config, "home": home, "serial": serial, "urdf": str(urdf)}


class _Rig:  # pragma: no cover - real hardware
    """Leader-to-follower teleoperation plus an aligned RealSense stream."""

    def __init__(self, settings: dict[str, Any], max_step_deg: float):
        import pyrealsense2 as rs
        from _so101_motion import _connected_driver
        from inspect_robots_so101 import packing
        from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader

        self._packing = packing
        config = replace(
            settings["config"], camera_configs={}, cameras=(), max_relative_target=max_step_deg
        )
        home = settings["home"]
        self.follower = _connected_driver(
            config, arm_position_p_coefficient=home.get("arm_position_p_coefficient")
        )
        self.leader = SOLeader(
            SO101LeaderConfig(
                port=home["leader_port"],
                id=home["leader_id"],
                calibration_dir=(
                    Path(home["leader_calibration_dir"])
                    if home.get("leader_calibration_dir")
                    else None
                ),
                use_degrees=True,
            )
        )
        self.leader.connect(calibrate=False)
        rs_config = rs.config()
        rs_config.enable_device(settings["serial"])
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(rs_config)
        self._align = rs.align(rs.stream.color)
        self._scale = profile.get_device().first_depth_sensor().get_depth_scale()
        k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = np.array([[k.fx, 0, k.ppx], [0, k.fy, k.ppy], [0, 0, 1.0]])

    def joints(self) -> np.ndarray:
        return np.asarray(self._packing.from_obs_dict(self.follower.get_observation()))

    def leader_joints(self) -> np.ndarray:
        return np.asarray(self._packing.from_obs_dict(self.leader.get_action()))

    def follow(self) -> None:
        self.follower.send_action(self.leader.get_action())

    def frames(self) -> tuple[np.ndarray, np.ndarray]:
        aligned = self._align.process(self._pipeline.wait_for_frames())
        rgb = np.asanyarray(aligned.get_color_frame().get_data()).copy()
        depth = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float32)
        return rgb, depth * np.float32(self._scale)

    def close(self) -> None:
        for close in (self._pipeline.stop, self.leader.disconnect, self.follower.disconnect):
            try:
                close()
            except Exception as exc:  # keep closing the rest
                print(f"cleanup: {exc}", file=sys.stderr)


def _settle(rig: _Rig) -> np.ndarray:  # pragma: no cover - real hardware
    """Hold the follower on the leader target until it stops moving."""
    previous = rig.joints()
    for _ in range(40):
        rig.follow()
        time.sleep(0.05)
        current = rig.joints()
        if np.max(np.abs(current[:5] - previous[:5])) < SETTLE_DEG:
            return current
        previous = current
    raise RuntimeError("arm did not settle; hold the leader still and press SPACE again")


def _terminal_command() -> str | None:  # pragma: no cover - real terminal
    """Return one typed line without blocking, or None when nothing was entered."""
    import select

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.readline().strip().lower() if ready else None


def collect(args: argparse.Namespace) -> int:  # pragma: no cover - real hardware
    """Teleoperate, capture marker/joint pairs, then solve (or verify) interactively."""
    import rerun as rr
    from _so101_capx import load_extrinsics, load_kinematics

    settings = rig_settings(args.config)
    config = settings["config"]
    kinematics = load_kinematics(
        settings["urdf"], joint_low=config.joint_low, joint_high=config.joint_high
    )
    samples_path = Path(args.samples)
    samples = [] if args.fresh or args.verify else load_samples(samples_path)
    if samples:
        print(f"continuing with {len(samples)} saved captures (use --fresh to start over)")
    verify = None
    if args.verify:
        document = json.loads(Path(args.out).read_text(encoding="utf-8"))
        verify = (np.linalg.inv(load_extrinsics(args.out)), np.asarray(document["marker_offset_m"]))
    rr.init("so101_handeye", spawn=True)
    rig = _Rig(settings, args.max_step_deg)
    try:
        gap = float(np.max(np.abs(rig.leader_joints()[:5] - rig.joints()[:5])))
        print(f"leader/follower joint gap: {gap:.1f} deg")
        if gap > 15:
            answer = input(
                f"Large gap: the follower will move to the leader pose at "
                f"{args.max_step_deg:g} deg/step. Type yes to continue: "
            )
            if answer.strip().lower() != "yes":
                return 1
        print(COMMANDS_VERIFY if verify is not None else COMMANDS)
        while True:
            rig.follow()
            rgb, depth = rig.frames()
            corners = detect(rgb, args.marker_id)
            rr.log("camera", rr.Image(rgb).compress(jpeg_quality=80))
            rr.log(
                "camera/marker",
                rr.LineStrips2D(
                    [np.vstack([corners, corners[:1]])] if corners is not None else [],
                    colors=[0, 255, 0],
                    radii=2.0,
                ),
            )
            status = f"{len(samples)} samples" + (
                "" if corners is not None else " | marker NOT visible"
            )
            if verify is not None and corners is not None:
                base_to_camera, offset = verify
                tool = kinematics.forward_kinematics(rig.joints()[:5])
                predicted_base = tool[:3, :3] @ offset + tool[:3, 3]
                predicted = base_to_camera[:3, :3] @ predicted_base + base_to_camera[:3, 3]
                rr.log(
                    "camera/predicted",
                    rr.Points2D(
                        [project(predicted, rig.intrinsics)],
                        colors=[255, 0, 0],
                        radii=6.0,
                    ),
                )
                try:
                    measured, _ = marker_point(corners, depth, rig.intrinsics)
                    status = f"verify error {np.linalg.norm(measured - predicted) * 1000:.1f} mm"
                except ValueError as exc:
                    status = str(exc)
            rr.log("status", rr.TextDocument(status))
            command = _terminal_command()
            if command is None:
                continue
            if command == "x":
                print("aborted; captures so far remain in", samples_path)
                return 1
            if verify is not None:
                if command == "q":
                    return 0
                print(status)
                continue
            if command == "":
                try:
                    joints = _settle(rig)
                    rgb, depth = rig.frames()
                    corners = detect(rgb, args.marker_id)
                    if corners is None:
                        raise RuntimeError("marker not visible")
                    tracking = float(np.max(np.abs(rig.leader_joints()[:5] - joints[:5])))
                    if tracking > TRACKING_DEG:
                        raise RuntimeError(f"follower lags the leader by {tracking:.1f} deg")
                    point, fraction = marker_point(corners, depth, rig.intrinsics)
                except (RuntimeError, ValueError) as exc:
                    print(f"  not captured: {exc}")
                else:
                    samples.append(
                        {
                            "joints_deg": joints.tolist(),
                            "camera_point_m": point.tolist(),
                            "pixel": corners.mean(axis=0).tolist(),
                            "depth_fraction": fraction,
                        }
                    )
                    save_samples(samples_path, samples, camera_serial=settings["serial"])
                    print(
                        f"  captured #{len(samples) - 1}: camera point "
                        f"{np.round(point, 3).tolist()} m, depth {fraction:.0%}"
                    )
            elif command == "u":
                if samples:
                    samples.pop()
                    save_samples(samples_path, samples, camera_serial=settings["serial"])
                print(f"  {len(samples)} samples left")
            elif command == "q":
                if len(samples) < MIN_SAMPLES:
                    print(f"  need at least {MIN_SAMPLES} samples, have {len(samples)}")
                    continue
                print("Move the leader to its rest pose now; torque is released on exit.")
                input("Press Enter when the follower is resting... ")
                break
            else:
                print(f"  unknown command {command!r}")
    finally:
        rig.close()
    return finish(samples, kinematics, args, settings["serial"], settings["urdf"], rig.intrinsics)


def finish(
    samples: list[dict[str, Any]],
    kinematics: Any,
    args: argparse.Namespace,
    serial: str | None,
    urdf: str,
    intrinsics: np.ndarray | None,
) -> int:
    """Solve the kept samples, print the report, and write the extrinsics file."""
    dropped = sorted({int(item) for item in (args.drop or "").split(",") if item.strip()})
    kept = [sample for index, sample in enumerate(samples) if index not in dropped]
    if len(kept) < MIN_SAMPLES:
        print(f"need at least {MIN_SAMPLES} samples, have {len(kept)}", file=sys.stderr)
        return 1
    solution = solve_samples(kept, kinematics)
    print(report(solution, dropped))
    document = result_document(
        solution,
        camera_serial=serial,
        urdf=urdf,
        samples_file=str(Path(args.samples)),
        dropped=dropped,
        intrinsics=None if intrinsics is None else np.asarray(intrinsics).tolist(),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if solution.rms <= WARN_RMS_M else 2


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wiring
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(_REPO / ".so101/capx.ini"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=2.0,
        help="follower per-command joint limit while teleoperating",
    )
    parser.add_argument("--print-marker", metavar="PNG", help="write a printable marker and exit")
    parser.add_argument("--fresh", action="store_true", help="ignore earlier saved captures")
    parser.add_argument("--solve-only", action="store_true", help="re-solve saved captures")
    parser.add_argument("--drop", default="", help="comma-separated capture indices to ignore")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="live check of an existing result; shows the error in mm",
    )
    args = parser.parse_args(argv)
    if args.print_marker:
        write_marker(Path(args.print_marker), args.marker_id)
        print(f"wrote {args.print_marker}: print it so the black square is 3-4 cm wide")
        return 0
    if args.solve_only:
        from _so101_capx import load_kinematics

        settings = rig_settings(args.config)
        config = settings["config"]
        kinematics = load_kinematics(
            settings["urdf"], joint_low=config.joint_low, joint_high=config.joint_high
        )
        data = json.loads(Path(args.samples).read_text(encoding="utf-8"))
        return finish(
            list(data["samples"]),
            kinematics,
            args,
            data.get("camera_serial"),
            settings["urdf"],
            None,
        )
    return collect(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
