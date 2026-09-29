"""Read the placement MJCF source without importing a physics or rendering backend."""
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


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


def read_scene_xml(path=None):
    source = (Path(path).resolve() if path is not None else
              Path(str(files('vision_arm_lab').joinpath('simulation/assets/placement.xml'))))
    root = parse_scene_xml(source.read_text())
    compiler = root.find('compiler')
    if compiler is not None and any(key in compiler.attrib for key in ('assetdir', 'meshdir', 'texturedir', 'strippath')):
        raise ValueError('Use asset file paths relative to scene XML; compiler asset directories are unsupported')
    # Store a resolved snapshot in SceneConfig so reset/replay never rereads a
    # possibly edited source file. Built-in assets are resolved by the backend.
    for node in root.findall('asset/*[@file]'):
        value = node.get('file')
        if not value.startswith('robosuite://'):
            node.set('file', str((source.parent / value).resolve()))
    return ET.tostring(root, encoding='unicode')


def scene_parameters(xml, camera_name):
    root = parse_scene_xml(xml)
    if root.tag != 'mujoco' or root.find('.//include') is not None:
        raise ValueError('Scene must be a single <mujoco> document without <include>')

    def element(path):
        node = root.find(path)
        if node is None:
            raise ValueError(f'Scene XML is missing {path}')
        return node

    def vector(node, attribute, count):
        try:
            values = np.array([float(v) for v in node.attrib[attribute].split()])
        except (KeyError, ValueError) as exc:
            raise ValueError(f'Scene XML {node.get("name", node.tag)} requires numeric {attribute}') from exc
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f'Scene XML {attribute} requires {count} finite values')
        return values

    table = element("worldbody/body[@name='table']")
    table_geom = element("worldbody/body[@name='table']/geom[@name='table_collision']")
    table_visual = element("worldbody/body[@name='table']/geom[@name='table_visual']")
    cube = element("worldbody/body[@name='cube_main']")
    cube_geom = element("worldbody/body[@name='cube_main']/geom[@name='cube_g0']")
    cube_visual = element("worldbody/body[@name='cube_main']/geom[@name='cube_g0_vis']")
    target = element("worldbody/site[@name='target_region']")
    camera = element(f"worldbody/camera[@name='{camera_name}']")
    table_half = vector(table_geom, 'size', 3)
    cube_half = vector(cube_geom, 'size', 3)
    # The placement evaluator and sampler require an axis-aligned, centered
    # table and a single cube. Reject shapes they cannot evaluate correctly.
    for body in (table, cube):
        if any(key in body.attrib for key in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis')):
            raise ValueError('Table and cube body orientation must be set by the placement task')
    for geom, visual in ((table_geom, table_visual), (cube_geom, cube_visual)):
        if geom.get('type') != 'box' or visual.get('type') != 'box':
            raise ValueError('Table and cube must remain box geoms')
        if not np.array_equal(vector(geom, 'size', 3), vector(visual, 'size', 3)):
            raise ValueError('Collision and visual box sizes must match in scene XML')
        for node in (geom, visual):
            if np.any(vector(node, 'pos', 3)) or any(key in node.attrib for key in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis')):
                raise ValueError('Table and cube geoms must stay centered and axis-aligned')
    table_pos = vector(table, 'pos', 3)
    if np.any(table_pos[:2]) or not np.all(cube_half == cube_half[0]):
        raise ValueError('Placement requires a table centered at x=y=0 and an equal-sided cube')
    if np.any(table_half <= 0) or np.any(cube_half <= 0):
        raise ValueError('Table and cube sizes must be positive')
    if target.get('type') != 'box' or any(key in target.attrib for key in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis')):
        raise ValueError('target_region must remain an axis-aligned box site')
    if cube.find("joint[@name='cube_joint0'][@type='free']") is None:
        raise ValueError('cube_joint0 must remain a free joint')
    if cube.find('inertial') is not None or cube.find('body') is not None or len(cube.findall('geom')) != 2:
        raise ValueError('Cube mass must come from its collision geom; retain its two geoms')
    if float(cube_visual.get('mass', 'nan')) != 1e-8:
        raise ValueError('Keep cube visual geom mass at 1e-8; edit cube_g0 mass or density')
    try:
        mass = (float(cube_geom.get('mass')) if 'mass' in cube_geom.attrib else
                float(cube_geom.attrib['density']) * float(np.prod(2 * cube_half)))
        timestep = float(element('option').attrib['timestep'])
        samples = int(element('visual/quality').attrib['offsamples'])
        fovy = float(camera.attrib['fovy'])
    except (KeyError, ValueError) as exc:
        raise ValueError('Scene requires explicit cube mass/density, timestep, offsamples and camera fovy') from exc
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError('Scene timestep must be finite and positive')
    return {
        'table_size_m': (2 * table_half).tolist(),
        'table_height_m': float(table_pos[2] + table_half[2]),
        'friction': vector(table_geom, 'friction', 3).tolist(),
        'cube_side_m': float(2 * cube_half[0]), 'cube_mass_kg': mass,
        'target_center_m': vector(target, 'pos', 3)[:2].tolist(),
        'target_size_m': (2 * vector(target, 'size', 3)[:2]).tolist(),
        'camera_position_m': vector(camera, 'pos', 3).tolist(),
        'camera_quaternion_wxyz': vector(camera, 'quat', 4).tolist(),
        'camera_fovy_deg': fovy, 'physics_hz': 1 / timestep,
        'offscreen_samples': samples,
    }
