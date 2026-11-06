"""Install the pinned official SO-101 URDF and meshes before a robot run.

Runtime kinematics only reads the local cache. This script is the sole network
step; it refuses a changed upstream URDF and never overwrites an existing model.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPOSITORY = "https://github.com/TheRobotStudio/SO-ARM100.git"
REVISION = "5f6d2b876a53a4872e405b991dd925556c9e38a4"
URDF_SHA256 = "3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c"
SOURCE = Path("Simulation/SO101")
DEFAULT_DEST = Path(__file__).resolve().parent.parent / ".so101/models/SO101"


def _check(path: Path) -> set[Path]:
    urdf = path / "so101_new_calib.urdf"
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
    if digest != URDF_SHA256:
        raise ValueError(f"SO-101 URDF checksum mismatch at {urdf}: {digest}")
    root = ET.parse(urdf).getroot()
    assets = {Path(mesh.attrib["filename"]) for mesh in root.findall(".//mesh")}
    if any(asset.is_absolute() or ".." in asset.parts for asset in assets):
        raise ValueError("SO-101 URDF contains an unsafe mesh path")
    for asset in assets:
        if not (path / asset).is_file():
            raise FileNotFoundError(path / asset)
    return assets


def install(destination: Path = DEFAULT_DEST) -> Path:
    """Populate an absent cache atomically from the pinned source revision."""
    destination = destination.expanduser().resolve()
    if destination.exists():
        _check(destination)
        return destination / "so101_new_calib.urdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="so101-model-", dir=destination.parent) as temp:
        temp_path = Path(temp)
        checkout = temp_path / "checkout"
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "remote", "add", "origin", REPOSITORY], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "fetch",
                "-q",
                "--depth",
                "1",
                "--filter=blob:none",
                "origin",
                REVISION,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "sparse-checkout", "set", str(SOURCE)], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "advice.detachedHead=false",
                "checkout",
                "-q",
                "FETCH_HEAD",
            ],
            check=True,
        )
        source = checkout / SOURCE
        assets = _check(source)
        staged = temp_path / "SO101"
        staged.mkdir()
        shutil.copy2(source / "so101_new_calib.urdf", staged)
        for asset in assets:
            (staged / asset).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / asset, staged / asset)
        shutil.copy2(checkout / "LICENSE", staged / "UPSTREAM_LICENSE")
        _check(staged)
        staged.rename(destination)
    return destination / "so101_new_calib.urdf"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    print(install(args.dest))
