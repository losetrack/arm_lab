"""The single assembly point for the supported Panda placement environment."""
from vision_arm_lab.simulation.control import PandaActionAdapter
from vision_arm_lab.core.contracts import ActionSpec, EnvironmentSpec


def make_backend(simulation, episode_timeout_s, *, render_mode='offscreen'):
    if render_mode not in ('offscreen', 'window'):
        raise ValueError('render_mode must be offscreen or window')
    # Heavy simulator imports are deferred until a real environment is requested.
    from vision_arm_lab.simulation.robosuite import RobosuiteBackend, panda_controller_config
    from vision_arm_lab.simulation.scene import CubePlacement
    from vision_arm_lab.simulation.placement_state import read_placement_state

    window = render_mode == 'window'

    def scene_factory(seed):
        return CubePlacement(
            scene_config=simulation, robots='Panda',
            controller_configs=panda_controller_config(),
            has_renderer=window, has_offscreen_renderer=True,
            use_camera_obs=True, use_object_obs=False,
            camera_names=simulation.camera_name,
            camera_heights=simulation.camera_size_px[0],
            camera_widths=simulation.camera_size_px[1], camera_depths=True,
            control_freq=simulation.control_hz,
            horizon=round(episode_timeout_s * simulation.control_hz),
            initialization_noise=None, seed=seed, hard_reset=False,
        )

    backend = RobosuiteBackend(
        scene_factory, camera_name=simulation.camera_name,
        adapter=PandaActionAdapter(simulation.grasp_quaternion_xyzw), window=window,
    )
    return backend, lambda: read_placement_state(backend.env)


def describe_environment(simulation, task_info, episode_timeout_s):
    return EnvironmentSpec(
        action_spec=ActionSpec(), camera_names=(simulation.camera_name,),
        camera_size_px=simulation.camera_size_px,
        joint_names=tuple(f'robot0_joint{i}' for i in range(1, 8)),
        capabilities=frozenset({'rgb', 'depth', 'calibration', 'robot_state', 'language'}),
        task=task_info, episode_timeout_s=episode_timeout_s,
    )


def make_expert_locator(environment):
    """Explicit privileged adapter; never supplied to normal policy factories."""
    return lambda observation: environment._read_task_state().position_m
