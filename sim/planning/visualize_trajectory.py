"""
visualize_trajectory.py

Animates a generated target trajectory inside the actual MuJoCo scene, so you
can SEE the planned path (as waypoint markers) and the live motion together,
against the real obstacle geometry -- not just the abstract (x, y) numbers
trajectory_generator.py prints.

This is purely kinematic playback: the target's freejoint qpos is set
directly each frame from the time-parameterized trajectory samples, and we
never call mj_step -- so gravity/physics never acts on the target or the
drone. This is intentional for a first visual sanity check; it is NOT a
substitute for actually driving the target with a real physics-based
controller during episode collection later.

Usage (run from project root or scripts/, matching visualize_scene.py):
    python scripts/visualize_trajectory.py --scene warehouse
    python scripts/visualize_trajectory.py --scene courtyard --speed 1.2 --waypoints 8
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim.planning.trajectory_generator import TrajectoryConfig, generate_episode_trajectory, generate_waypoints  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.parent
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
    return f"""<mujoco model="following_env_{scene_name}_trajectory_preview">
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
 
    # Same fix as visualize_scene.py: x2.xml's <compiler assetdir="assets"/>
    # resolves relative to wherever the merged file lives, so write the temp
    # file inside mujoco_menagerie/skydio_x2/ for that to resolve correctly.
    menagerie_x2_dir = PROJECT_ROOT / "mujoco_menagerie" / "skydio_x2"
    if not menagerie_x2_dir.exists():
        raise FileNotFoundError(f"Expected mujoco_menagerie/skydio_x2/ at {menagerie_x2_dir}")
 
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
 
 
def _draw_trajectory_markers(
    viewer,
    traj: list[tuple[float, float, float]],
    waypoints: list[tuple[float, float]],
    z: float = 0.05,
) -> None:
    """Draws small markers along the ENTIRE trajectory (not just waypoints),
    striding the samples so the marker count stays under the viewer's
    maxgeom limit. Waypoints are drawn larger/red for reference; the full
    path (now a smoothed spline when config.smooth=True, matching what the
    target actually follows) is drawn as smaller cyan dots.
    """
    scn = viewer.user_scn
    scn.ngeom = 0
 
    # Reserve room for waypoint markers, stride the rest across the budget.
    budget = scn.maxgeom - len(waypoints) - 1
    stride = max(1, len(traj) // max(1, budget))
 
    for i in range(0, len(traj), stride):
        if scn.ngeom >= scn.maxgeom:
            break
        x, y = traj[i][0], traj[i][1]
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.035, 0, 0],
            pos=[x, y, z],
            mat=[1, 0, 0, 0, 1, 0, 0, 0, 1],
            rgba=[0.2, 0.8, 1.0, 0.6],
        )
        scn.ngeom += 1
 
    for (x, y) in waypoints:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.08, 0, 0],
            pos=[x, y, z],
            mat=[1, 0, 0, 0, 1, 0, 0, 0, 1],
            rgba=[1, 0.2, 0.2, 0.9],
        )
        scn.ngeom += 1
 
 
def main():
    parser = argparse.ArgumentParser(description="Animate a generated target trajectory in a scene")
    parser.add_argument("--scene", choices=list(SCENE_FILES.keys()), default="warehouse")
    parser.add_argument("--waypoints", type=int, default=6)
    parser.add_argument("--speed", type=float, default=1.5, help="target speed, m/s")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loop", action="store_true", help="loop the trajectory forever")
    args = parser.parse_args()
 
    if args.scene == "urban":
        print(
            "[warning] urban is currently excluded from Phase 1 data collection "
            "(occlusion risk, see scene_loader.py) -- visualizing anyway for "
            "inspection purposes only."
        )
 
    config = TrajectoryConfig(
        num_waypoints=args.waypoints, target_speed=args.speed, seed=args.seed
    )
 
    print(f"Generating trajectory for scene: {args.scene}")
    waypoints = generate_waypoints(args.scene, config)
    traj = generate_episode_trajectory(args.scene, config)
    print(f"Generated {len(traj)} samples over {traj[-1][2]:.2f}s "
          f"across {len(waypoints)} waypoints")
 
    model = load_model(args.scene)
    data = mujoco.MjData(model)
 
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ground_target")
    if target_body_id == -1:
        raise RuntimeError("Could not find body 'ground_target' in the composed model")
    qpos_adr = model.jnt_qposadr[model.body_jntadr[target_body_id]]
 
    # Hold target at fixed height (its authored pos z=0.02) and identity
    # orientation throughout -- only x, y are driven by the trajectory.
    target_z = model.body_pos[target_body_id][2]
 
    mujoco.mj_forward(model, data)
 
    with mujoco.viewer.launch_passive(model, data) as viewer:
        _draw_trajectory_markers(viewer, traj, waypoints, z=target_z + 0.03)
 
        start_wall = time.time()
        while viewer.is_running():
            elapsed = time.time() - start_wall
            t = elapsed % traj[-1][2] if args.loop else min(elapsed, traj[-1][2])
 
            # Find the sample at or after t (linear scan is fine at this scale;
            # trajectories here run to at most a few hundred samples).
            sample = traj[-1]
            for s in traj:
                if s[2] >= t:
                    sample = s
                    break
 
            data.qpos[qpos_adr : qpos_adr + 3] = [sample[0], sample[1], target_z]
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]  # identity quat
            data.qvel[:] = 0  # no physics integration; pure kinematic override
 
            mujoco.mj_forward(model, data)
            viewer.sync()
 
            if not args.loop and elapsed > traj[-1][2] + 1.0:
                break
 
            time.sleep(1.0 / 60.0)
 
 
if __name__ == "__main__":
    main()
 