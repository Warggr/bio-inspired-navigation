# coding: utf-8

# This script creates a binary occupancy map based on one or more .urdf files containing all walls.
# For map layouts that use .obj files, rather than .urdf, use create_occupancy_map_from_obj.py instead.

import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import sys, os


def parse_coords(coords: str):
    return tuple(float(i) for i in coords.split(" "))


img_root = "system/controller/simulation/environment/"
folder = sys.argv[1]

if len(sys.argv) > 2:
    filename = sys.argv[2]
else:
    filename = "maze_topview_binary.png"
outfile = os.path.join(img_root, folder, filename)

files = ['plane.urdf'] + sys.argv[3:]

starts, stops = [], []

for file in files:
    filename = os.path.join(img_root, folder, file)
    tree = ET.parse(filename)
    root = tree.getroot()

    links = [el for el in root if el.tag == "link"]
    joints = [el for el in root if el.tag == "joint"]

    children = dict()
    for joint in joints:
        assert joint.find("parent").attrib["link"] == "planeLink"
        origin = parse_coords(joint.find("origin").attrib["xyz"])
        children[joint.find("child").attrib["link"]] = origin

    for i, link in enumerate(links):
        if link.attrib["name"] == "planeLink":
            continue
        origin = children[link.attrib["name"]]
        link = link.find("collision")
        obj_origin = link.find("origin").attrib["xyz"]
        size = link.find("geometry").find("box").attrib["size"]
        obj_origin, size = parse_coords(obj_origin), parse_coords(size)
        origin = np.array(obj_origin) + np.array(origin)
        #assert origin[2] == 0 and size[2] == 1, (origin, size)
        origin, size = origin[:2], np.array(size[:2])
        starts.append(origin - size / 2)
        stops.append(origin + size / 2)

starts = np.array(starts, dtype=float)
stops = np.array(stops, dtype=float)

resolution = 0.05
corner = np.min(starts, axis=0)
starts = ((starts - corner) / resolution).astype(int)
stops = ((stops - corner) / resolution).astype(int)
dims = np.max(stops, axis=0)


image = Image.new(mode="1", size=tuple(dims), color=(1))

draw = ImageDraw.Draw(image)
for start, stop in zip(starts, stops):
    print("Drawing rectangle", start, stop)
    draw.rectangle([tuple(start), tuple(stop)], fill=0)

image = ImageOps.flip(image)
image.save(outfile)
