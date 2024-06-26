# coding: utf-8
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_, in_name, out_name, title = sys.argv

#data = np.genfromtxt('data/results/autoencoder_performance.csv', delimiter=',')
data = pd.read_csv(in_name)
x, y = data.keys()
plt.plot(data[x], data[y])
plt.title(title)
plt.xlabel(x)
plt.ylabel(y)
plt.savefig(out_name, format='png')
