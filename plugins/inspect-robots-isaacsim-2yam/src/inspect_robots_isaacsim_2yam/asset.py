"""Prepare MolmoAct2's YAM MJCF for Isaac Sim without editing source assets."""

from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PreparedMjcf:
    """An Isaac-compatible MJCF copy and its temporary-directory owner."""

    path: Path
    _temporary_directory: tempfile.TemporaryDirectory[str]

    def cleanup(self) -> None:
        """Remove the private MJCF and mesh copies."""
        self._temporary_directory.cleanup()


def prepare_isaac_mjcf(source: Path) -> PreparedMjcf:
    """Copy and normalize a flattened YAM MJCF for Isaac's importer.

    MolmoAct2 names separate left and right mesh assets but points both names at
    the same OBJ files. Isaac Sim converts OBJ files beside their source and can
    race when two mesh assets share one path. The private copy gives right-arm
    meshes unique paths and makes inherited geometry types explicit.
    """
    temporary_directory = tempfile.TemporaryDirectory(prefix="inspect-robots-2yam-")
    root = Path(temporary_directory.name)
    try:
        tree = ET.parse(source)
        compiler = tree.getroot().find("compiler")
        mesh_dir_name = compiler.get("meshdir", "") if compiler is not None else ""
        source_mesh_dir = (source.parent / mesh_dir_name).resolve()
        target_mesh_dir = root / "assets"
        if source_mesh_dir.is_dir():
            shutil.copytree(
                source_mesh_dir,
                target_mesh_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*_tmp"),
            )
        else:
            target_mesh_dir.mkdir()
        if compiler is not None:
            compiler.set("meshdir", "assets")

        base = tree.getroot().find("./worldbody/body[@name='bimanual_base']")
        if base is not None and base.find("inertial") is None:
            # Give Isaac's fixed articulation root valid mass properties. The
            # source wrapper is intentionally empty because MuJoCo does not
            # require inertia on a fixed body.
            base.insert(
                0,
                ET.Element(
                    "inertial",
                    pos="0 0 0",
                    mass="0.1",
                    diaginertia="0.001 0.001 0.001",
                ),
            )

        for mesh in tree.getroot().findall("./asset/mesh"):
            name = mesh.get("name", "")
            file_name = mesh.get("file")
            if not name.startswith("right_") or file_name is None:
                continue
            source_mesh = source_mesh_dir / file_name
            if not source_mesh.is_file():
                raise RuntimeError(f"YAM mesh asset was not found: {source_mesh}")
            unique_name = f"{name}{source_mesh.suffix}"
            shutil.copy2(source_mesh, target_mesh_dir / unique_name)
            mesh.set("file", unique_name)

        for geom in tree.getroot().iter("geom"):
            if geom.get("type") is not None:
                continue
            if geom.get("mesh") is not None:
                geom.set("type", "mesh")
            elif geom.get("class", "").endswith("_collision"):
                geom.set("type", "capsule")

        output = root / source.name
        tree.write(output, encoding="utf-8", xml_declaration=True)
        return PreparedMjcf(path=output, _temporary_directory=temporary_directory)
    except Exception:
        temporary_directory.cleanup()
        raise
