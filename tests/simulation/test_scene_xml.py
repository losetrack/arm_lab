from dataclasses import asdict, replace
import json
import xml.etree.ElementTree as ET

import pytest
import yaml

from vision_arm_lab.simulation.config import SceneConfig, load_config


def test_relative_xml_and_derived_task_priors(custom_scene, monkeypatch):
    config_file, xml_file, tree = custom_scene
    tree.find("worldbody/site[@name='target_region']").set('pos', '0.14 0.16 0.8005')
    tree.find("worldbody/camera[@name='agentview']").set('fovy', '55')
    tree.write(xml_file)
    monkeypatch.chdir('/')
    config = load_config(config_file)
    assert config.spec.task.target_center_m == (0.14, 0.16)
    assert config.placement.target_center_m == (0.14, 0.16)
    assert config.camera_fovy_deg == 55
    assert config.cube_side_m == 0.04
    assert config.cube_mass_kg == pytest.approx(0.1)
    assert config.physics_hz == 500


def test_snapshot_survives_source_removal_and_metadata_roundtrip(custom_scene):
    config_file, xml_file, _ = custom_scene
    config = load_config(config_file)
    xml_file.unlink()
    restored = SceneConfig(**asdict(config))
    assert restored == config
    assert load_config(restored) is restored
    with pytest.raises(FileNotFoundError):
        load_config(config_file)
    with pytest.raises(ValueError, match='must match scene XML'):
        replace(config, cube_side_m=0.06)


def test_configuration_vectors_are_owned_immutable_snapshots(custom_scene):
    config = load_config(custom_scene[0])
    values = json.loads(json.dumps(asdict(config)))
    restored = SceneConfig(**values)
    simulation, spec, placement = restored.simulation, restored.spec, restored.placement
    for name, value in values.items():
        if isinstance(value, list):
            value[0] += 1
            assert getattr(restored, name) == getattr(config, name)
            with pytest.raises(TypeError):
                getattr(restored, name)[0] = value[0]
    assert simulation.target_center_m == spec.task.target_center_m == placement.target_center_m
    assert SceneConfig(**restored.record_metadata()['scene']) == config


def test_duplicate_physical_yaml_parameter_is_rejected(custom_scene):
    config_file, _, _ = custom_scene
    values = yaml.safe_load(config_file.read_text())
    values['cube_side_m'] = 0.06
    config_file.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='Physical parameters belong in scene XML'):
        load_config(config_file)


@pytest.mark.parametrize('change', ['missing_target', 'size_mismatch', 'non_cube', 'rotated_table', 'include'])
def test_unsupported_scene_geometry_fails_before_simulation(custom_scene, change):
    config_file, xml_file, tree = custom_scene
    root = tree.getroot()
    if change == 'missing_target':
        root.find('worldbody').remove(root.find("worldbody/site[@name='target_region']"))
    elif change == 'size_mismatch':
        root.find("worldbody/body[@name='cube_main']/geom[@name='cube_g0']").set('size', '0.03 0.03 0.03')
    elif change == 'non_cube':
        for geom in root.findall("worldbody/body[@name='cube_main']/geom"):
            geom.set('size', '0.03 0.02 0.02')
    elif change == 'rotated_table':
        root.find("worldbody/body[@name='table']").set('euler', '0.1 0 0')
    else:
        ET.SubElement(root, 'include', file='unresolved.xml')
    tree.write(xml_file)
    with pytest.raises(ValueError):
        load_config(config_file)


def test_custom_asset_path_is_relative_to_xml(custom_scene):
    config_file, xml_file, tree = custom_scene
    tree.find("asset/texture[@name='texplane']").set('file', 'textures/floor.png')
    tree.write(xml_file)
    snapshot = ET.fromstring(load_config(config_file).scene_xml)
    assert snapshot.find("asset/texture[@name='texplane']").get('file') == str(xml_file.parent / 'textures/floor.png')


def test_malformed_xml_is_a_configuration_error(custom_scene):
    config_file, xml_file, _ = custom_scene
    xml_file.write_text('<mujoco>')
    with pytest.raises(ValueError, match='Invalid scene XML'):
        load_config(config_file)
