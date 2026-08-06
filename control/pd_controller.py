import numpy as np
import mujoco

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def quat_to_yaw(quat):
    w, x, y, z = quat
    return np.arctan2(2*(w*z + x*y), 1- 2*(y*y + z*z))

def quat_to_roll_pitch(quat):
    w, x, y, z = quat
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    return roll, pitch

class PDLandingController:
    def __init__(self,
                 k_xy=0.5, kd_xy=0.1, vmax_xy=1.0,
                 k_z = 0.5, kd_z=0.1, vmax_z=0.7,
                 k_yaw=1.2, max_yaw_rate=0.8, yaw_deadband=0.05,
                 k_level=2.0, max_level_rate=1.0,
                #  descend_rate=0.3, center_radius=0.5,
                landing_height=0.05,
                 alpha=0.7):
        # gains, mirroring your PX4 node's parameter names directly
        self.k_xy = k_xy
        self.kd_xy = kd_xy
        self.vmax_xy = vmax_xy

        self.k_z = k_z
        self.kd_z = kd_z
        self.vmax_z = vmax_z

        self.k_yaw = k_yaw
        self.max_yaw_rate = max_yaw_rate
        self.yaw_deadband = yaw_deadband

        self.k_level = k_level
        self.max_level_rate = max_level_rate

        # self.descend_rate = descend_rate
        # self.center_radius = center_radius

        self.landing_height = landing_height
        self.alpha = alpha  # D-term smoothing factor

        self.reset()

    def reset(self):
        # error history, needed for finite-difference D term
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_ez = 0.0
        self.dex_f = 0.0
        self.dey_f = 0.0
        self.dez_f = 0.0

    def compute(self, relative_pos, relative_quat, dt):
        ex, ey, ez_raw = relative_pos   # drone - marker
        ez = ez_raw - self.landing_height  # error relative to landing height
        print('ez_raw is ', ez_raw, 'self.landing_height is ', self.landing_height, 'ez is ', ez)

        # --- D term: filtered finite difference ---
        dex = (ex - self.prev_ex) / dt
        dey = (ey - self.prev_ey) / dt
        dez = (ez - self.prev_ez) / dt
        self.dex_f = self.alpha * self.dex_f + (1 - self.alpha) * dex
        self.dey_f = self.alpha * self.dey_f + (1 - self.alpha) * dey
        self.dez_f = self.alpha * self.dez_f + (1 - self.alpha) * dez
        self.prev_ex, self.prev_ey, self.prev_ez = ex, ey, ez

        #  --- PD combine for x/y, negative sign: error is (drone - marker)
        # so to move drone TOWARD marker we command velocity opposite the error ---
        vx = (self.k_xy * ex + self.kd_xy * self.dex_f)
        vy = (self.k_xy * ey + self.kd_xy * self.dey_f)
        vz = (self.k_z * ez + self.kd_z * self.dez_f)
        vx = np.clip(vx, -self.vmax_xy, self.vmax_xy)
        vy = np.clip(vy, -self.vmax_xy, self.vmax_xy)
        if(abs(ez) > 3.0):
            vz = np.clip(vz, -self.vmax_z, self.vmax_z)
            if((abs(ex) > 1 or abs(ey) > 1)):
                vz = 0.3 * vz
        else:
            vz = -self.vmax_z * (1 - np.e ** (ez * 2))
            if((abs(ex) > 1 or abs(ey) > 1)):
                vz = 0.0

        # --- yaw: unchanged, P-only ---
        yaw_err = wrap_to_pi(quat_to_yaw(relative_quat))
        if abs(yaw_err) < self.yaw_deadband:
            yaw_rate = 0.0
        else:
            yaw_rate = np.clip(self.k_yaw * yaw_err, -self.max_yaw_rate, self.max_yaw_rate)

        return np.array([vx, vy, vz, yaw_rate])

    def compute_level_correction(self, drone_quat):
        roll, pitch = quat_to_roll_pitch(drone_quat)
        roll_rate = np.clip(-self.k_level * roll, -self.max_level_rate, self.max_level_rate)
        pitch_rate = np.clip(-self.k_level * pitch, -self.max_level_rate, self.max_level_rate)
        return roll_rate, pitch_rate


