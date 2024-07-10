# coding: utf-8
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import sys, os

img_root = "system/controller/simulation/environment/"
folder = sys.argv[1]

filename = os.path.join(img_root, folder, "plane.urdf")
tree = ET.parse(filename)
root = tree.getroot()

links = [el for el in root if el.tag == "link"]
joints = [el for el in root if el.tag == "joint"]

children = set()
for joint in joints:
    assert joint.find("parent").attrib["link"] == "planeLink"
    assert joint.find("origin").attrib["xyz"] == "0 0 0"
    children.add(joint.find("child").attrib["link"])

starts = np.zeros((len(links), 2), dtype=float)
stops = np.zeros((len(links), 2), dtype=float)

def parse_coords(coords: str):
    return np.array([float(i) for i in coords.split(" ")])

for i, link in enumerate(links):
    if link.attrib["name"] == "planeLink":
        continue
    else:
        assert link.attrib["name"] in children
    link = link.find("collision")
    origin = link.find("origin").attrib["xyz"]
    size = link.find("geometry").find("box").attrib["size"]
    origin, size = parse_coords(origin), parse_coords(size)
    assert origin[2] == 0 and size[2] == 1
    origin, size = origin[:2], size[:2]
    starts[i] = origin - size / 2
    stops[i] = origin + size / 2

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
image.save(os.path.join(img_root, folder, "maze_topview_binary.png"))
