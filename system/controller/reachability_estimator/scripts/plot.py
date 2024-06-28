# coding: utf-8
import argparse
import sys
import matplotlib.pyplot as plt
import pandas as pd

_, in_name, out_name, title = sys.argv

data = pd.read_csv(in_name)
x, y = data.keys()
plt.scatter(data[x], data[y])
plt.title(title)
plt.xlabel(x)
plt.ylabel(y)
plt.savefig(out_name, format='png')
