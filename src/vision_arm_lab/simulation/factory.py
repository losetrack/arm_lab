"""The single assembly point for the supported Panda placement environment."""
from vision_arm_lab.simulation.config import load_config
from vision_arm_lab.simulation.control import PandaActionAdapter
from vision_arm_lab.core.environment import Environment
from vision_arm_lab.tasks.placement import PlacementEvaluator
from vision_arm_lab.simulation.scene_xml import scene_asset_fingerprint


def make_environment(config, *, render_mode='offscreen') -> Environment:
    config = load_config(config)
    if render_mode not in ('offscreen', 'window'):
        raise ValueError('render_mode must be offscreen or window')
    # Heavy simulator imports are deferred until a real environment is requested.
    from vision_arm_lab.simulation.robosuite import RobosuiteBackend, panda_controller_config
    from vision_arm_lab.simulation.scene import CubePlacement
    from vision_arm_lab.simulation.placement_state import read_placement_state

    simulation = config.simulation
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
            horizon=round(config.episode_timeout_s * simulation.control_hz),
            initialization_noise=None, seed=seed, hard_reset=False,
        )

    backend = RobosuiteBackend(
        scene_factory, camera_name=simulation.camera_name,
        adapter=PandaActionAdapter(simulation.grasp_quaternion_xyzw), window=window,
    )
    return Environment(
        backend, PlacementEvaluator(config.placement),
        lambda: read_placement_state(backend.env), config.spec, render_mode=render_mode,
    )


def make_recorded_environment(metadata, *, render_mode='offscreen') -> Environment:
    """Rebuild a recorded scene using its snapshot, after checking its assets."""
    from vision_arm_lab.simulation.config import SceneConfig

    snapshot = metadata['scene']
    if metadata['scene_asset_sha256'] != scene_asset_fingerprint(snapshot['scene_xml']):
        raise ValueError('Replay requires the recorded scene asset versions')
    return make_environment(SceneConfig(**snapshot), render_mode=render_mode)


def make_builtin_policy(name, environment):
    from vision_arm_lab.algorithms.policies import GraspPolicy, HoldPolicy
    from vision_arm_lab.algorithms.perception import ColorLocator

    if name == 'inspect':
        return HoldPolicy(environment.spec)
    if name == 'expert':
        # Only this explicitly privileged assembly path receives the state reader.
        locate = lambda observation: environment._read_task_state().position_m
    elif name == 'vision':
        locate = ColorLocator(environment.spec.task)
    else:
        raise ValueError(f'Unknown built-in policy: {name}')
    return GraspPolicy(environment.spec.task, locate)
