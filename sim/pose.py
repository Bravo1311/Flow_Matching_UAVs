import mujoco
import numpy as np

def get_marker_id(model, marker_name="landing_marker"):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, marker_name)

def get_relative_pose(model, data, marker_id):
    drone_pos = data.qpos[0:3].copy()  # copying freezes the value at the moment
    marker_pose = data.xpos[marker_id].copy()
    relative_pos = marker_pose - drone_pos
    return relative_pos

def get_relative_orientation(data, marker_id):
    drone_quat = data.qpos[3:7].copy()
    marker_quat = data.xquat[marker_id].copy()

    marker_quat_inv = np.zeros(4)
    # inverse of a unit quarternion = negate the vector part
    mujoco.mju_negQuat(marker_quat_inv, marker_quat)  

    relative_quat = np.zeros(4)
    # marker_inv * drone
    mujoco.mju_mulQuat(relative_quat, marker_quat_inv, drone_quat)
    return relative_quat

def random_yaw_quat():
    yaw_angle = np.random.uniform(-np.pi, np.pi)  # random yaw, full circle
    axis = np.array([0.0, 0.0, 1.0])               # z-axis = yaw
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, yaw_angle)
    return quat
                          