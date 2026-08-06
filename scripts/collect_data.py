import argparse
import time
import mujoco.viewer
import os
import json
import numpy as np
import mujoco
from control.pd_controller import PDLandingController
from sim.pose import get_marker_id, get_relative_orientation, get_relative_pose

#  ---- Config ----
MODEL_PATH = "mujoco_menagerie/skydio_x2/scene.xml"
DATA_DIR = "data/episodes"
NUM_EPISODES = 1000
MAX_EPISODE_TIME = 20.0
SUCCESS_HOLD_STEPS = 20
XY_TOL = 0.1
Z_TOL = 0.05
TAU = 0.3
LANDING_HEIGHT = 0.05 

# Randomization ranges for spawn (relative to marker at origin)
SPAWN_XY_RANGE = (-6.0, 6.0)
SPAWN_Z_RANGE = (4.0, 6.0)

os.makedirs(DATA_DIR, exist_ok=True)

def random_yaw_quat():
    yaw_angle = np.random.uniform(-np.pi, np.pi)
    axis = np.array([0.0, 0.0, 1.0])
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, yaw_angle)
    return quat

def quat_to_roll_pitch(quat):
    w, x, y, z = quat
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    return roll, pitch

def randomize_spawn(data, model):
    x = np.random.uniform(*SPAWN_XY_RANGE)
    y = np.random.uniform(*SPAWN_XY_RANGE)
    z = np.random.uniform(*SPAWN_Z_RANGE)
    data.qpos[0:3] = [x, y, z]
    data.qpos[3:7] = random_yaw_quat()   # level, random yaw only
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

def run_episode(model, data, marker_id, controller, ep_idx):
    randomize_spawn(data, model)
    controller.reset()
    v_actual = np.zeros(4)
    dt = model.opt.timestep

    log = []
    success_counter = 0
    t = 0.0

    while t < MAX_EPISODE_TIME:
        rel_pos = get_relative_pose(model, data, marker_id)
        rel_quat = get_relative_orientation(data, marker_id)
        cmd_vel = controller.compute(rel_pos, rel_quat, dt)

        # levelling correction, independent of the 4D action policy will learn
        roll_rate, pitch_rate = controller.compute_level_correction(data.qpos[3:7])

        log.append({
            "t": t,
            "relative_pos": rel_pos.tolist(),
            "relative_quat": rel_quat.tolist(),
            "cmd_vel": cmd_vel.tolist(), # not the policy output
        })

        v_actual += (cmd_vel - v_actual) * (dt / TAU)
        data.qvel[0:3] = v_actual[0:3]
        data.qvel[3] = roll_rate
        data.qvel[4] = pitch_rate
        data.qvel[5] = v_actual[3]

        mujoco.mj_step(model, data)
        t += dt

        ex, ey, ez_raw = rel_pos
        if abs(ex) < XY_TOL and abs(ey) < XY_TOL and abs(ez_raw - LANDING_HEIGHT) < Z_TOL:
            success_counter += 1
            if success_counter >= SUCCESS_HOLD_STEPS:
                return log, "success"
        else:
            success_counter = 0

    return log, "timeout"

def run_episode_visual(model, data, marker_id, controller, viewer):
    """Same as run_episode, but renders live and paces to real time. Does not save."""
    randomize_spawn(data, model)
    controller.reset()
    v_actual = np.zeros(4)
    dt = model.opt.timestep

    success_counter = 0
    t = 0.0

    while t < MAX_EPISODE_TIME and viewer.is_running():
        step_start = time.time()

        rel_pos = get_relative_pose(model, data, marker_id)
        rel_quat = get_relative_orientation(data, marker_id)
        cmd_vel = controller.compute(rel_pos, rel_quat, dt)
        roll_rate, pitch_rate = controller.compute_level_correction(data.qpos[3:7])

        v_actual += (cmd_vel - v_actual) * (dt / TAU)
        data.qvel[0:3] = v_actual[0:3]
        data.qvel[3] = roll_rate
        data.qvel[4] = pitch_rate
        data.qvel[5] = v_actual[3]

        mujoco.mj_step(model, data)
        viewer.sync()
        t += dt

        ex, ey, ez_raw = rel_pos
        if abs(ex) < XY_TOL and abs(ey) < XY_TOL and abs(ez_raw - LANDING_HEIGHT) < Z_TOL:
            success_counter += 1
            if success_counter >= SUCCESS_HOLD_STEPS:
                return "success"
        else:
            success_counter = 0

        elapsed = time.time() - step_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    return "timeout"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", type=int, default=0,
                         help="If >0, run this many episodes with the viewer instead of collecting data")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    marker_id = get_marker_id(model)
    controller = PDLandingController(landing_height=LANDING_HEIGHT)

    if args.visualize > 0:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for i in range(args.visualize):
                outcome = run_episode_visual(model, data, marker_id, controller, viewer)
                print(f"Episode {i+1}/{args.visualize}: {outcome}")
        return

    # --- normal headless collection path, unchanged ---
    num_success = 0
    for ep_idx in range(NUM_EPISODES):
        log, outcome = run_episode(model, data, marker_id, controller, ep_idx)
        if outcome == "success":
            num_success += 1
            out_path = os.path.join(DATA_DIR, f"episode_{ep_idx:04d}.json")
            with open(out_path, "w") as f:
                json.dump({"outcome": outcome, "steps": log}, f)
        if (ep_idx + 1) % 20 == 0:
            print(f"[{ep_idx+1}/{NUM_EPISODES}] success rate so far: {num_success}/{ep_idx+1}")
    print(f"Done. {num_success}/{NUM_EPISODES} episodes succeeded and were saved.")


if __name__ == "__main__":
    main()