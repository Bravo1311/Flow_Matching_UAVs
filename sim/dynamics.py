import time
import numpy as np
import mujoco
import mujoco.viewer
from pynput import keyboard
from sim.keyboard_teleop import TeleopState
from sim.pose import *
from control.pd_controller import PDLandingController

def quat_to_euler(quat):
    w, x, y, z = quat
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.degrees([roll, pitch, yaw])

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/skydio_x2/scene.xml")
# model = mujoco.MjModel.from_xml_path("assets/scene.xml")
model.opt.gravity[:] = [0, 0, -9.81] 
G = 9.81
HOVER_BIAS = G * model.opt.timestep  # per-step compensation term

marker_id = get_marker_id(model)

data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
data.ctrl[:] = 0.0
data.qpos[0:3] = [7.0, 2.0, 5.0]  # start 1m up, clear of ground
q = np.array(random_yaw_quat())
q = q / np.linalg.norm(q)
data.qpos[3:7] = q
mujoco.mj_forward(model, data)

euler = np.zeros(3)
mujoco.mju_quat2Vel(euler, q, 1.0) 

controller = PDLandingController()

teleop = TeleopState()
listener = keyboard.Listener(on_press=teleop.on_press, on_release=teleop.on_release)
listener.start()

TAU = 0.3  # lag time constant (seconds) — smaller = snappier
v_actual = np.zeros(4)  # [vx, vy, vz, yaw_rate], current actual velocity

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        dt = model.opt.timestep

        # teleop.update_cmd_vel()
        # v_actual += (teleop.cmd_vel - v_actual) * (dt / TAU)
        rel_pos = get_relative_pose(model, data, marker_id)
        rel_quat = get_relative_orientation(data, marker_id)
        cmd_vel = controller.compute(rel_pos, rel_quat, dt)

        v_actual += (cmd_vel - v_actual) * (dt/TAU)

        # v_actual[2] += HOVER_BIAS  # counteract gravity's per-step pull, on top of commanded/lagged vz

        data.qvel[0:3] = v_actual[0:3]  # linear velocity, world frame
        data.qvel[2] += HOVER_BIAS  # linear velocity, world frame
        data.qvel[5] = v_actual[3]      # yaw rate (angular vel about z)

        mujoco.mj_step(model, data)
        viewer.sync()

        print("roll, pitch, yaw (deg):", quat_to_euler(q))

        roll_rate, pitch_rate = controller.compute_level_correction(data.qpos[3:7])

        data.qvel[0:3] = v_actual[0:3]
        data.qvel[3] = roll_rate
        data.qvel[4] = pitch_rate
        data.qvel[5] = v_actual[3]


        if int(data.time * 10) % 10 == 0:  # print roughly once per second
            # print(f"t={data.time:.1f}s  altitude={data.qpos[2]:.4f}m")
            rel_pos = get_relative_pose(model, data, marker_id)
            # print(f"t={data.time:.1f}s  relative_pos={rel_pos}")
            rel_quat = get_relative_orientation(data, marker_id)
            # print(f"t={data.time:.1f}s  relative_quat={rel_quat}")

        elapsed = time.time() - step_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)