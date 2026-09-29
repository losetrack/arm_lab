"""Editing source XML must affect the simulator and the public task priors."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from vision_arm_lab import make_environment
from vision_arm_lab.configuration import load_config

pytestmark = pytest.mark.integration


def test_edited_xml_is_used_without_python_geometry_overrides(custom_scene):
    config_file, xml_file, tree = custom_scene
    root = tree.getroot()
    cube = root.find("worldbody/body[@name='cube_main']")
    for geom in cube.findall('geom'):
        geom.set('size', '0.025 0.025 0.025')
    cube.find("geom[@name='cube_g0']").set('mass', '0.2')
    root.find("worldbody/camera[@name='agentview']").set('fovy', '55')
    root.find("worldbody/site[@name='target_region']").set('pos', '0.14 0.16 0.8005')
    obstacle = ET.SubElement(root.find('worldbody'), 'body', name='custom_obstacle', pos='0.32 0.32 0.84')
    ET.SubElement(obstacle, 'geom', name='custom_obstacle_geom', type='box', size='0.02 0.02 0.04', group='1')
    tree.write(xml_file)
    config = load_config(config_file)
    xml_file.unlink()  # Reset must use the loaded snapshot, including custom geoms.
    with make_environment(config) as environment:
        observation = environment.reset(0)
        env = environment._backend.env
        assert environment.spec.task.cube_side_m == 0.05
        assert environment.spec.task.target_center_m == (0.14, 0.16)
        assert env.sim.model.body_mass[env.cube_body_id] == pytest.approx(0.2)
        np.testing.assert_allclose(env.sim.model.geom_size[env.sim.model.geom_name2id('cube_g0')], [0.025]*3)
        assert env.sim.model.cam_fovy[env.sim.model.camera_name2id('agentview')] == 55
        assert 'custom_obstacle_geom' in env.sim.model.geom_names
        assert observation.cameras['agentview'].rgb.shape == (256, 256, 3)
