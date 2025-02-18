import mujoco
import time


def view():
    xml_path = "./mujoco_plane/resources/labyrinth_wo_meshes_actuators.xml"
    
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
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("Viewer closed. Exiting Mujoco visualization.")