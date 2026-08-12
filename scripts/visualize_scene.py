"""
visualize_scene.py

Standalone sanity-check viewer for the Phase 1 scenes. Deliberately bypasses
scene_loader.py's caching/randomization for now -- the goal here is just to
confirm each scene XML composes and renders correctly with the real x2.xml
robot body and target_body.xml, before adding per-episode randomization on
top.

Usage:
    python visualize_scene.py --scene warehouse
    python visualize_scene.py --scene courtyard
    python visualize_scene.py --scene urban

Run from the flow_matching_following/ project root.
"""

import argparse
import tempfile
from pathlib import Path

import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).parent.parent  # visualize_scene.py lives in sim/, so go up one more level to the actual repo root (containing mujoco_menagerie/, sim/, control/, etc.)
SCENES_DIR = PROJECT_ROOT / "sim" / "scenes"
ROBOT_INCLUDE = PROJECT_ROOT / "sim" / "assets" / "skydio_x2.xml"
TARGET_INCLUDE = PROJECT_ROOT / "sim" / "assets" / "target_body.xml"

SCENE_FILES = {
    "warehouse": SCENES_DIR / "scene_warehouse.xml",
    "courtyard": SCENES_DIR / "scene_courtyard.xml",
    "urban": SCENES_DIR / "scene_urban.xml",
}


def compose_mjcf(scene_name: str) -> str:
    scene_path = SCENE_FILES[scene_name]
    return f"""<mujoco model="following_env_{scene_name}_preview">
  <include file="{scene_path.as_posix()}"/>
  <include file="{ROBOT_INCLUDE.as_posix()}"/>
  <include file="{TARGET_INCLUDE.as_posix()}"/>
</mujoco>
"""


def load_model(scene_name: str) -> mujoco.MjModel:
    for label, path in [
        ("scene", SCENE_FILES.get(scene_name)),
        ("robot include", ROBOT_INCLUDE),
        ("target include", TARGET_INCLUDE),
    ]:
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    mjcf_str = compose_mjcf(scene_name)

    # MJCF <include> paths resolve relative to CWD when loading from a
    # string, not relative to this script's location. Writing to a temp
    # file in the project root and loading via from_xml_path sidesteps
    # that ambiguity entirely -- more robust than relying on CWD.
    # Critical: x2.xml has <compiler assetdir="assets"/>, a path RELATIVE to
    # wherever the final merged/compiled XML file lives -- NOT relative to
    # x2.xml's own original directory. <include> splices XML text together
    # before any path resolution happens, so if we write the composed temp
    # file to the project root, MuJoCo looks for `assets/X2_lowpoly.obj`
    # next to the temp file and fails. Writing the temp file inside
    # mujoco_menagerie/skydio_x2/ instead makes that relative path resolve
    # correctly again. Our scene/robot/target <include> paths are already
    # absolute (via .as_posix() on absolute Path objects), so moving the
    # temp file's location doesn't break those.
    menagerie_x2_dir = PROJECT_ROOT / "mujoco_menagerie" / "skydio_x2"
    if not menagerie_x2_dir.exists():
        raise FileNotFoundError(
            f"Expected mujoco_menagerie/skydio_x2/ at {menagerie_x2_dir}"
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=menagerie_x2_dir, delete=False
    ) as f:
        f.write(mjcf_str)
        temp_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return model


def main():
    parser = argparse.ArgumentParser(description="Visualize a Phase 1 following-policy scene")
    parser.add_argument(
        "--scene", choices=list(SCENE_FILES.keys()), default="warehouse",
        help="Which scene to load (default: warehouse)"
    )
    args = parser.parse_args()

    print(f"Loading scene: {args.scene}")
    model = load_model(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Loaded OK -- nbody={model.nbody} ngeom={model.ngeom} "
          f"nlight={model.nlight} nmat={model.nmat}")
    print("Opening viewer (close window to exit)...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()