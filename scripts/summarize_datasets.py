# coding: utf-8
from pathlib import Path
import h5py

p = Path('system/controller/reachability_estimator/data/reachability')
datasets = p.glob('dataset*.hd5*')
datasets = filter(lambda f : not f.is_symlink(), datasets)
for d in datasets:
    f = h5py.File(d)
    dlength = len(f['positions'])
    dattrs = ','.join(map(lambda keyval: f"{keyval[0]}={keyval[1]}", f.attrs.items()))
    print(f"{d}:\t{dlength}\t{dattrs}")
