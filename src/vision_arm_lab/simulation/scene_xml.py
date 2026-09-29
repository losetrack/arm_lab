"""Read the placement MJCF source without importing a physics or rendering backend."""
from hashlib import sha256
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path
import xml.etree.ElementTree as ET



def parse_scene_xml(xml):
    try:
        return ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError(f'Invalid scene XML: {exc}') from exc


def resolve_asset_file(value):
    if value.startswith('robosuite://'):
        package = Path(find_spec('robosuite').origin).parent
        return package / 'models/assets' / value.removeprefix('robosuite://')
    return Path(value)


def scene_asset_fingerprint(xml):
    return {node.get('file'): sha256(resolve_asset_file(node.get('file')).read_bytes()).hexdigest()
            for node in ET.fromstring(xml).findall('asset/*[@file]')}


def read_scene_xml(path=None):
    source = (Path(path).resolve() if path is not None else
              Path(str(files('vision_arm_lab').joinpath('simulation/assets/placement.xml'))))
    root = parse_scene_xml(source.read_text())
    compiler = root.find('compiler')
    if compiler is not None and any(key in compiler.attrib for key in ('assetdir', 'meshdir', 'texturedir', 'strippath')):
        raise ValueError('Use asset file paths relative to scene XML; compiler asset directories are unsupported')
    # Store a resolved snapshot in the simulation config so reset/replay never rereads a
    # possibly edited source file. Built-in assets are resolved by the backend.
    for node in root.findall('asset/*[@file]'):
        value = node.get('file')
        if not value.startswith('robosuite://'):
            node.set('file', str((source.parent / value).resolve()))
    return ET.tostring(root, encoding='unicode')
