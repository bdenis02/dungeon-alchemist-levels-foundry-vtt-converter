import os
import json
import argparse
import re
import random

import PIL
from PIL import Image, ImageDraw
from networkx import Graph, DiGraph, simple_cycles
from bidict import bidict
from math import isclose, sqrt

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help')


parser.add_argument('filename')
parser.add_argument('-f', '--ground_floor', type=int, default=0)
parser.add_argument('-s', '--floor_height', type=int, default=10)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--tile_save_path', type=str)

"""
to add to lights to make them torches

        "animation": {
          "type": "torch",
          "speed": 5,
          "intensity": 5,
          "reverse": false
        }
"""

args = parser.parse_args()
random.seed(args.random_seed)
abspath = os.path.abspath(args.filename)
basepath = os.path.basename(abspath)
dirpath = os.path.dirname(abspath)

PIL.Image.MAX_IMAGE_PIXELS = 1_000_000_000

if basepath.endswith('.json') or basepath.endswith('.jpg'):
    basepath = '_'.join(basepath.split('_')[:-1])

if args.tile_save_path is None:
    args.tile_save_path = f'assets/scenes/{basepath}'


def get_paths(strings, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_\d{{2}}\.(jpg|json)$")
    return [s for s in strings if pattern.match(s)]


filenames = get_paths(os.listdir(dirpath), basepath)
image_filenames = sorted([i for i in filenames if i.endswith('jpg')])
json_filenames = sorted([i for i in filenames if i.endswith('json')])
assert len(image_filenames) == len(json_filenames)
assert all([i[-6:-4] == j[-7:-5]
           for i, j in zip(image_filenames, json_filenames)])

args.number_of_levels = len(image_filenames)


ENVIRONMENT = {
    'darknessLevel': 0,
    'darknessLock': False,
    'globalLight': {
        'enabled': True,
        'alpha': 0.5,
        'bright': False,
        'color': None,
        'coloration': 1,
        'luminosity': 0,
        'saturation': 0,
        'contrast': 0,
        'shadows': 0,
        'darkness': {
            'min': 0,
            'max': 0.5}},
    'cycle': True,
    'base': {
        'hue': 0,
        'intensity': 0,
        'luminosity': 0,
        'saturation': 0,
        'shadows': 0},
    'dark': {
        'hue': 0.7138888888888889,
        'intensity': 0,
        'luminosity': -0.25,
        'saturation': 0,
        'shadows': 0}}


def convert_map(
        dirpath,
        basename,
        tile_path,
        json_filenames,
        image_filenames,
        ground_floor_index,
        floor_height):
    output_json = None
    previous_floor_polygons = None
    total_floors = len(json_filenames)
    print(json_filenames)
    print(image_filenames)
    for current_floor_index, (jsonfile, imagefile) in enumerate(
            zip(json_filenames, image_filenames)):
        print(current_floor_index)
        print(jsonfile)
        print(imagefile)

        current_floor_polygons = create_floor_polygons(dirpath, jsonfile)

        output_json = update_json(
            output_json,
            dirpath,
            basename,
            tile_path,
            current_floor_index,
            imagefile,
            jsonfile,
            ground_floor_index,
            total_floors,
            floor_height,
            current_floor_polygons)

        # print(f'{len(current_floor_polygons)=}')
        # for poly in current_floor_polygons:
        #    print(poly)
        mask_polygons = previous_floor_polygons + \
            current_floor_polygons if previous_floor_polygons is not None else current_floor_polygons
        create_image(
            dirpath,
            imagefile,
            mask_polygons,
            get_below_or_ground(
                ground_floor_index,
                current_floor_index))
        previous_floor_polygons = current_floor_polygons

    with open(os.path.join(dirpath, f'{basename}_converted.json'), 'w', encoding='utf-8') as json_out:
        json.dump(output_json, json_out, indent=4)
        # create_json(dirpath, json_filenames, ground_floor_index, floor_height)
        # create_images(dirpath, json_filenames, ddd)

# Json stuff


def update_json(
        output_json,
        dirpath,
        basename,
        tile_path,
        current_floor_index,
        imagefile,
        jsonfile,
        ground_floor_index,
        total_floors,
        floor_height,
        current_floor_polygons):
    json_path = os.path.join(dirpath, jsonfile)
    if output_json is None:
        with open(json_path, encoding='utf-8') as f:
            output_json = json.load(f)
        del output_json['walls']
        del output_json['lights']
        output_json['environment'] = ENVIRONMENT
        print(basename)
        output_json['name'] = basename
        output_json['grid'] = create_grid(output_json)
        output_json['padding'] = 0.0
        output_json['flags'] = create_flags(
            total_floors, ground_floor_index, floor_height)
    output_json = output_json.copy()
    current_floor = current_floor_index - ground_floor_index
    bottom_height = floor_height * current_floor
    top_height = floor_height * (current_floor + 1)

    with open(json_path, encoding='utf-8') as f:
        current_json = json.load(f)

    current_json['walls']
    # for wall in current_json['walls']:
    # print(wall)
    new_walls = [
        wall | {
            'flags': {
                'wall-height': {
                    'bottom': bottom_height,
                    'top': top_height}}} for wall in current_json['walls']]
    output_json['walls'] = new_walls if 'walls' not in output_json else output_json['walls'] + new_walls

    # for wall in output_json['walls']:
    # print(wall)

    # for light in current_json['lights']:
    # print(light)

    new_lights = [light | {'elevation': bottom_height,
                           'config': {
                               "animation": {
                                   "type": "torch",
                                   "speed": 5,
                                   "intensity": 5,
                                   "reverse": False
                               }
                           },
                           'flags': {
                               'levels': {
                                   'rangeTop': top_height
                               },
                               "tagger": {
                                   'tags': ""
                               }
                           }
                           } for light in current_json['lights']]
    output_json['lights'] = new_lights if 'lights' not in output_json else output_json['lights'] + new_lights

    # for light in current_json['lights']:
    # print(light)
    # for p in current_floor_polygons:
    #    print(p)
    new_region = {'name': f'Inside Buildings {bottom_height} - {top_height}',
                  'color': "#" + ''.join([random.choice('0123456789abcdef') for j in range(6)]),
                  'shapes': [{'type': 'polygon',
                              'points': [i for j in polygon for i in j]} for polygon in current_floor_polygons],
                  'hole': False,
                  'elevation': {'bottom': bottom_height,
                                'top': top_height - 1},
                  "behaviors": [{"name": "Suppress Weather",
                                 "type": "suppressWeather",
                                 "system": {},
                                 "disabled": False,
                                 "flags": {}},
                                {"name": "Adjust Darkness Level",
                                 "type": "adjustDarknessLevel",
                                 "system": {"mode": 0,
                                            "modifier": 1},
                                 "disabled": False,
                                 "flags": {}}],
                  "visibility": 0,
                  "locked": False,
                  "flags": {"tagger": {"tags": ""}}}
    output_json['regions'] = [
        new_region] if 'regions' not in output_json else output_json['regions'] + [new_region]
    image_path = os.path.join(dirpath, imagefile)
    img = Image.open(image_path)
    new_tile = {
        "texture": {
            "src": tile_path + '/' + get_image_output_file(imagefile),
            "anchorX": 0.5,
            "anchorY": 0.5,
            "offsetX": 0,
            "offsetY": 0,
            "fit": "fill",
            "scaleX": 1,
            "scaleY": 1,
            "rotation": 0,
            "tint": "#ffffff",
            "alphaThreshold": 0.75
        },
        "x": 0,
        "y": 0,
        "width": img.width,
        "height": img.height,
        "elevation": bottom_height,
        "sort": 0,
        "occlusion": {
            "mode": 1,
            "alpha": 0
        },
        "rotation": 0,
        "alpha": 1,
        "hidden": False,
        "locked": False,
        "restrictions": {
            "light": False,
            "weather": False
        },
        "video": {
            "loop": True,
            "autoplay": True,
            "volume": 0
        },
        "flags": {
            "betterroofs": {
                "brMode": False
            },
            "levels": {
                "rangeTop": 0,
                "allWallBlockSight": False
            }
        }
    }
    output_json['tiles'] = [
        new_tile] if 'tiles' not in output_json else output_json['tiles'] + [new_tile]

    return output_json


def create_grid(initial_json):
    return {
        'type': 1,
        'size': initial_json['grid'],
        'style': 'solidLines',
        'thickness': 1,
        'color': '#000000',
        'alpha': 0.5,
        'distance': initial_json['gridDistance'],
        'units': initial_json['gridUnits']}


def create_flags(number_of_levels, zero_floor, floor_height):
    return {'levels': {'lightMasking': False,
                       'sceneLevels': [[floor_height * i,
                                        floor_height * (i + 1),
                                        f'level {i}'] for i in range(-zero_floor,
                                                                     number_of_levels - zero_floor)],
                       'wall-height': {'advancedVision': True}}}


# Floor polygon related stuff

def create_floor_polygons(dirpath, jsonfile):
    with open(os.path.join(dirpath, jsonfile), encoding='utf-8') as f:
        map_json = json.load(f)
    walls = [((wall['c'][0], wall['c'][1]), (wall['c'][2], wall['c'][3]))
             for wall in map_json['walls']]

    # Prepare edge/points/dictionaries/etc
    pointdict = bidict()
    edgesdict = {}
    pointindex = 0
    for wall in walls:
        p0 = wall[0]
        if p0 not in pointdict:
            pointdict[p0] = pointindex
            pointindex += 1
        p1 = wall[1]
        if p1 not in pointdict:
            pointdict[p1] = pointindex
            pointindex += 1

        p0index = pointdict[p0]
        p1index = pointdict[p1]

        if p0index not in edgesdict:
            edgesdict[p0index] = set([p1index,])
        else:
            edgesdict[p0index].add(p1index)

        if p1index not in edgesdict:
            edgesdict[p1index] = set([p0index,])
        else:
            edgesdict[p1index].add(p0index)
    G = Graph(edgesdict)
    DG = DiGraph(G)
    cycles = list(simple_cycles(DG))
    points_cycles = [[pointdict.inverse[point] for point in cycle]
                     for cycle in cycles if len(cycle) > 2]
    points_cycles = [convert_node_to_edge_poly(
        cycle) for cycle in points_cycles]
    points_cycles = remove_polygons_inside(points_cycles)
    return [convert_edge_to_node_poly(cycle) for cycle in points_cycles]


def remove_polygons_inside(cycles):
    cycles = cycles.copy()
    # Remove polygons that are entirely inside of another polygon
    index = 0
    while index < len(cycles):
        increment = True
        for i in range(index + 1, len(cycles)):
            # print(f'{points_cycles[i]=}')
            if all_points_inside(cycles[index], cycles[i]):
                # print(len(cycles))
                del cycles[index]
                # print(len(cycles))
                increment = False
                break
        if increment:
            index += 1
    return cycles


def convert_node_to_edge_poly(cycle):
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def convert_edge_to_node_poly(cycle):
    return [edge[0] for edge in cycle]


def all_points_inside(test_cycle, boundary_cycle):
    # print(f'{boundary_cycle=}')
    return all([is_inside(boundary_cycle, point[0][0], point[0][1]) and is_inside(
        boundary_cycle, point[1][0], point[1][1]) for point in test_cycle])


def edge_length(point0, point1):
    result = sqrt((point0[0] - point1[0])**2 + (point0[1] - point1[1])**2)
    return result


def is_on_edge(edge0, edge1, xp, yp):
    if (edge0[0] == xp and edge0[1] == yp) or (
            edge1[0] == xp and edge1[1] == yp):
        return True
    edge_ab = edge_length(edge0, (xp, yp))
    edge_bc = edge_length(edge1, (xp, yp))
    edge_ac = edge_length(edge0, edge1)
    length_abc = edge_ab + edge_bc
    result = isclose(length_abc, edge_ac, abs_tol=1e-8)
    return result


def is_inside(edges, xp, yp):
    cnt = 0
    for edge in edges:
        # print(edge)
        if is_on_edge(edge[0], edge[1], xp, yp):
            # print('is_on_egde')
            return True
        # print('not_on_edge')
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + \
                ((yp - y1) / (y2 - y1)) * (x2 - x1):
            cnt += 1
    return cnt % 2 == 1


# image processing related stuff
def get_below_or_ground(ground_floor_index, current_floor_index):
    return current_floor_index <= ground_floor_index


def get_image_output_file(imagefile):
    print(f'{imagefile=}')
    return f'{".".join(imagefile.split(".")[:-1])}.webp'


def create_image(dirpath, imagefile, mask_polygons, is_below_or_ground):
    # print(len(mask_polygons))
    # print(len(mask_polygons[0]))
    # print(len(mask_polygons[0][0]))
    # for poly in mask_polygons:
    # print('===')
    # for point in poly:
    #    print(point)
    out_file = get_image_output_file(imagefile)
    print(f'{dirpath=}')
    print(f'{out_file=}')
    out_path = os.path.join(dirpath, out_file)
    image_path = os.path.join(dirpath, imagefile)
    img = Image.open(image_path).convert("RGBA")

    if is_below_or_ground:
        img.save(out_path)
        return
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)

    for poly in mask_polygons:
        # print(poly)
        draw.polygon(poly, fill=255)

    img.putalpha(mask)
    # print(out_path)
    img.save(out_path)


convert_map(
    dirpath,
    basepath,
    args.tile_save_path,
    json_filenames,
    image_filenames,
    args.ground_floor,
    args.floor_height)
