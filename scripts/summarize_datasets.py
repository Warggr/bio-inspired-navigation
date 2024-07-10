# coding: utf-8
from pathlib import Path
import h5py
import numpy as np
import sys
from tqdm import tqdm

if len(sys.argv) > 1:
    datasets = sys.argv[1:]
else:
    p = Path('system/controller/reachability_estimator/data/reachability')
    datasets = p.glob('dataset*.hd5*')
    datasets = filter(lambda f: not f.is_symlink(), datasets)
for d in datasets:
    try:
        print(f"{d}:", end='\t')
        f = h5py.File(d)
        dset = f['positions']
        dlength = len(dset)
        dattrs = ','.join(map(lambda keyval: f"{keyval[0]}={keyval[1]}", f.attrs.items()))
        print(f"size {dlength}\t{dattrs}")
        BLOCK_SIZE=10000
        reached = 0
        for i in (bar := tqdm(range(0, len(dset), BLOCK_SIZE), file=sys.stdout)):
            n1 = dset[i:i+BLOCK_SIZE]
            reached += sum(n1['reached'])
            bar.set_description(f"  {i+BLOCK_SIZE} / {reached} reached ({reached/(i+BLOCK_SIZE):.1%})")
    except BlockingIOError:
        print("Couldn't open dataset")
    except OSError:
        print("Couldn't read dataset")
