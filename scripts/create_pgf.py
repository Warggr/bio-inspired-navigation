# coding: utf-8
import csv
import os
import matplotlib.pyplot as plt

def plot(filename, ax):
    label = filename[len('run-Aug05_21-03-.._lxhalle.stud.rbg.tum.de'):]
    label = label.removesuffix('-tag-Loss_Validation.csv')
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        reader = list(reader)
        x, y = ([row[key] for row in reader] for key in ('Step', 'Value'))
        x, y = list(map(int, x)), list(map(float, y))
        assert x == list(range(1, 26)), x
        ax.plot(x, y, label=label)

fig, ax = plt.subplots()
for filename in os.listdir("/tmp/logs"):
    plot(filename, ax)
    
fig.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation loss')
plt.show()
