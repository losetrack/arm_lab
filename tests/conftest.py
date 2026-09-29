"""Editable scene/config fixtures shared by configuration and simulator tests."""
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from vision_arm_lab.core.scene_xml import read_scene_xml


@pytest.fixture
def custom_scene(tmp_path):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/mvp.yaml').read_text())
    config['scene_xml'] = 'scene.xml'
    config_file = tmp_path / 'runtime.yaml'
    config_file.write_text(yaml.safe_dump(config))
    xml_file = tmp_path / 'scene.xml'
    xml_file.write_text(read_scene_xml())
    return config_file, xml_file, ET.parse(xml_file)
