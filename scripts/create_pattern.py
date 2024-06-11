#!/usr/bin/env python
# coding: utf-8
import sys
from PIL import Image
import random

def randcolor(precision=8):
    PRECISION = 2**precision
    return tuple(min(255, int(256 / PRECISION) * random.randint(0, PRECISION+1)) for _ in range(3))

try:
    imgsize = sys.argv[1]
    height, width = map(int, imgsize.split('x'))
    outfile = sys.argv[2]
    if len(sys.argv) == 4:
        random.seed(int(sys.argv[3]))

    if len(sys.argv) == 5:
        precision = int(sys.argv[4])
    else:
        precision = 2

except Exception as err:
    print(f"Error:", err, file=sys.stderr)
    print(f"Usage: {sys.argv[0]} <H>x<W> <outfile> [seed] [precision]", file=sys.stderr)
    sys.exit(1)

pattern = Image.new('RGB', (height, width))

for i in range(height):
    for j in range(width):
        pattern.putpixel((i, j), randcolor(precision=precision))

pattern.save(outfile)
