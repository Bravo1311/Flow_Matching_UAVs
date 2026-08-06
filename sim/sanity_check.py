import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/skydio_x2/scene.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
data.qpos[2] = 3.0  # override altitude to 5m, keep hover ctrl values
mujoco.mj_forward(model, data)  # recompute derived quantities after manual qpos edit

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        data.ctrl[0] -= 0.0000005
        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)