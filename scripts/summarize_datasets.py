# coding: utf-8
from pathlib import Path
import h5py
import numpy as np

p = Path('system/controller/reachability_estimator/data/reachability')
datasets = p.glob('dataset*.hd5*')
datasets = filter(lambda f : not f.is_symlink(), datasets)
for d in datasets:
    f = h5py.File(d)
    dset = f['positions']
    dlength = len(dset)
    n1 = np.zeros((len(dset),), dtype=dset.dtype)
    dset.read_direct(n1)
    reached = sum(n1['reached'])
    dattrs = ','.join(map(lambda keyval: f"{keyval[0]}={keyval[1]}", f.attrs.items()))
    print(f"{d}:\t{dlength} / {reached} reached ({reached/dlength:.1%})\t{dattrs}")
