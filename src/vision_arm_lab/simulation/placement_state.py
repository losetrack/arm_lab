"""Private simulator state reader for placement evaluation and the expert."""
from vision_arm_lab.tasks.placement import ObjectState


def read_placement_state(env):
    body_id = env.cube_body_id
    return ObjectState(
        position_m=env.sim.data.body_xpos[body_id].copy(),
        rotation=env.sim.data.body_xmat[body_id].reshape(3, 3).copy(),
        linear_velocity_m_s=env.sim.data.get_body_xvelp(env.cube.root_body).copy(),
        angular_velocity_rad_s=env.sim.data.get_body_xvelr(env.cube.root_body).copy(),
        gripper_contact=env.check_contact(env.robots[0].gripper['right'], env.cube),
    )
