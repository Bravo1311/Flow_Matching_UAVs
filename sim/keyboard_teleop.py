from pynput import keyboard
import numpy as np

class TeleopState:
    def __init__(self):
        self.cmd_vel = np.zeros(4)  # [vx, vy, vz, yaw_rate]
        self.pressed = set()

    def on_press(self, key):
        try:
            self.pressed.add(key.char)
        except AttributeError:
            pass  # special keys (shift, etc.) handled separately
        if key == keyboard.Key.shift:
            self.pressed.add('shift')
        if key == keyboard.Key.space:
            self.pressed.add('space')

    def on_release(self, key):
        try:
            self.pressed.discard(key.char)
        except AttributeError:
            pass
        if key == keyboard.Key.shift:
            self.pressed.discard('shift')
        if key == keyboard.Key.space:
            self.pressed.discard('space')

    def update_cmd_vel(self, max_speed=1.0, max_yaw_rate=1.5):
        vx = vy = vz = yaw = 0.0
        if 'w' in self.pressed: vx += max_speed
        if 's' in self.pressed: vx -= max_speed
        if 'a' in self.pressed: vy += max_speed
        if 'd' in self.pressed: vy -= max_speed
        if 'space' in self.pressed: vz += max_speed
        if 'shift' in self.pressed: vz -= max_speed
        if 'e' in self.pressed: yaw += max_yaw_rate
        if 'q' in self.pressed: yaw -= max_yaw_rate
        self.cmd_vel[:] = [vx, vy, vz, yaw]