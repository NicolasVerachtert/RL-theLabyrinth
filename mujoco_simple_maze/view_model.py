import mujoco
import mujoco.viewer
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def view():
    xml_path = "./mujoco_simple_maze/resources/mjc_simple_maze.xml"

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    n_steps = 5

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            step_start = time.time()
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - \
                (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

logger.info("Viewer closed. Exiting Mujoco visualization.")
