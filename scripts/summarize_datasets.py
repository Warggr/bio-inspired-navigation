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
        #n1 = np.zeros((BLOCK_SIZE,), dtype=dset.dtype)
        reached = 0
        for i in (bar := tqdm(range(0, len(dset), BLOCK_SIZE))):
            #dset.read_direct(n1, np.s_[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE])
            n1 = dset[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE]
            reached += sum(n1['reached'])
            bar.set_description(f"        {(i+1)*BLOCK_SIZE} / {reached} reached ({reached/dlength:.1%})")
        #print(f"\t{dlength} / {reached} reached ({reached/dlength:.1%})")
    except BlockingIOError:
        print("Couldn't open dataset")
    except OSError:
        print("Couldn't read dataset")
