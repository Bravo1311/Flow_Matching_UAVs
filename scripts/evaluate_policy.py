import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
import torch

from control.pd_controller import PDLandingController
from sim.pose import get_marker_id, get_relative_pose, get_relative_orientation
from model.flow_matching_v1.inference import *
from model.flow_matching_v1.config import *

# --------- CONFIG ----------
MODEL_PATH = "mujoco_menagerie/skydio_x2/scene.xml"
CHECKPOINT_PATH = "checkpoints/v1/model.pt"
NUM_EPISODES = 50
MAX_EPISODE_TIME = 20.0
SUCCESS_HOLD_STEPS = 20
XY_TOL = 0.1
Z_TOL = 0.05
TAU = 0.3
LANDING_HEIGHT = 0.05
STEPS_PER_CHUNK = 3

SPAWN_XY_RANGE = (-6.0, 6.0)
SPAWN_Z_RANGE = (4.0, 8.0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_yaw_quat():
    yaw_angle = np.random.uniform(-np.pi, np.pi)
    axis = np.array([0.0, 0.0, 1.0])
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, yaw_angle)
    return quat

def randomize_spawn(data, model):
    x = np.random.uniform(*SPAWN_XY_RANGE)
    y = np.random.uniform(*SPAWN_XY_RANGE)
    z = np.random.uniform(*SPAWN_Z_RANGE)
    data.qpos[0:3] = [x, y, z]
    data.qpos[3:7] = random_yaw_quat()
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

def _step_policy_common(data, marker_id, policy, level_controller,
                         history_buffer, v_actual, current_chunk, chunk_idx, steps_since_replan, dt):
    """Shared per-step logic used by both headless and visual episode runners."""
    rel_pos = get_relative_pose(model=None, data=data, marker_id=marker_id)  # placeholder, replaced below
    return rel_pos  # not used directly — logic is inlined in each runner below for clarity

def run_policy_episode(model, data, marker_id, policy, level_controller):
    randomize_spawn(data, model)
    v_actual = np.zeros(4)
    dt = model.opt.timestep

    history_buffer = []
    success_counter = 0
    t = 0.0
    steps_since_replan = STEPS_PER_CHUNK
    current_chunk = None
    chunk_idx = 0

    while t < MAX_EPISODE_TIME:
        rel_pos = get_relative_pose(model, data, marker_id)
        rel_quat = get_relative_orientation(data, marker_id)
        pose_vec = np.concatenate([rel_pos, rel_quat])

        history_buffer.append(pose_vec)
        if len(history_buffer) > HISTORY_LEN:
            history_buffer.pop(0)

        while len(history_buffer) < HISTORY_LEN:
            history_buffer.insert(0, history_buffer[0])

        if steps_since_replan >= STEPS_PER_CHUNK:
            # converts history_arr list to array
            history_arr = np.stack(history_buffer)
            current_chunk = generate_action_chunk(policy, history_arr, DEVICE)
            chunk_idx = 0
            steps_since_replan = 0

        cmd_vel = current_chunk[chunk_idx]
        chunk_idx += 1
        steps_since_replan += 1
        roll_rate, pitch_rate = level_controller.compute_level_correction(data.qpos[3:7])

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
                return "success"
        else:
            success_counter = 0

    return "timeout"

def run_policy_episode_visual(model, data, marker_id, policy, level_controller, viewer):
    randomize_spawn(data, model)
    v_actual = np.zeros(4)
    dt = model.opt.timestep

    history_buffer = []
    success_counter = 0
    t = 0.0
    steps_since_replan = STEPS_PER_CHUNK
    current_chunk = None
    chunk_idx = 0

    while t < MAX_EPISODE_TIME and viewer.is_running():
        step_start = time.time()

        rel_pos = get_relative_pose(model, data, marker_id)
        rel_quat = get_relative_orientation(data, marker_id)
        pose_vec = np.concatenate([rel_pos, rel_quat])

        history_buffer.append(pose_vec)
        if len(history_buffer) > HISTORY_LEN:
            history_buffer.pop(0)
        while len(history_buffer) < HISTORY_LEN:
            history_buffer.insert(0, history_buffer[0])

        if steps_since_replan >= STEPS_PER_CHUNK:
            history_arr = np.stack(history_buffer)
            current_chunk = generate_action_chunk(policy, history_arr, DEVICE)
            chunk_idx = 0
            steps_since_replan = 0

        cmd_vel = current_chunk[chunk_idx]
        chunk_idx += 1
        steps_since_replan += 1

        roll_rate, pitch_rate = level_controller.compute_level_correction(data.qpos[3:7])

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
                         help="If >0, run this many episodes with the viewer instead of a full eval batch")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    marker_id = get_marker_id(model)

    policy = load_policy(CHECKPOINT_PATH, DEVICE)
    level_controller = PDLandingController()

    if args.visualize > 0:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for i in range(args.visualize):
                outcome = run_policy_episode_visual(model, data, marker_id, policy, level_controller, viewer)
                print(f"Episode {i+1}/{args.visualize}: {outcome}")
        return

    num_success = 0
    for ep_idx in range(NUM_EPISODES):
        outcome = run_policy_episode(model, data, marker_id, policy, level_controller)
        if outcome == "success":
            num_success += 1
        print(f"Episode {ep_idx+1}/{NUM_EPISODES}: {outcome}  (success rate so far: {num_success}/{ep_idx+1})")

    print(f"\nFinal success rate: {num_success}/{NUM_EPISODES} ({100*num_success/NUM_EPISODES:.1f}%)")


if __name__ == "__main__":
    main()
