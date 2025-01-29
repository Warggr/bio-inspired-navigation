# Review

## Issues
1. Missing```bc_model/transformations.npz```
```bash
$ snakemake data/models/reachability_network-3colors.25c
...
[Wed Jan 29 17:49:03 2025]
localrule 8:
    input: data/reachability/dataset.hd5
    output: data/reachability/dataset-patterns.hd5
    jobid: 1
    reason: Missing output files: data/reachability/dataset-patterns.hd5
    wildcards: basename=dataset
    resources: tmpdir=/tmp

pybullet build time: Jan 12 2025 19:25:00
Traceback (most recent call last):
  File "/home/tp2/bio-inspired-navigation/system/controller/reachability_estimator/data_generation/clone_dataset.py", line 5, in <module>
    from system.controller.reachability_estimator.data_generation.dataset import DATASET_KEY, display_samples
  File "/home/tp2/Documents/bio-inspired-navigation/system/controller/reachability_estimator/data_generation/dataset.py", line 23, in <module>
    from system.controller.reachability_estimator.reachability_estimation import SimulationReachabilityEstimator
  File "/home/tp2/Documents/bio-inspired-navigation/system/controller/reachability_estimator/reachability_estimation.py", line 179, in <module>
    from .ReachabilityDataset import SampleConfig
  File "/home/tp2/Documents/bio-inspired-navigation/system/controller/reachability_estimator/ReachabilityDataset.py", line 23, in <module>
    boundaryCellEncoder = BoundaryCellNetwork.load()
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tp2/Documents/bio-inspired-navigation/system/bio_model/bc_network/bc_encoding.py", line 39, in load
    arrays = np.load(filename)
             ^^^^^^^^^^^^^^^^^
  File "/home/tp2/anaconda3/envs/bionav/lib/python3.12/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/tp2/Documents/bio-inspired-navigation/system/bio_model/data/bc_model/transformations.npz'
``` 
Consequently, this also leads to the failure of testing the RE network
``` bash
$ snakemake data/results/reachability_network-3colors-val.log
Assuming unrestricted shared filesystem usage.
host: tp2-Precision-3650-Tower
Building DAG of jobs...
MissingInputException in rule 13 in file /home/tp2/bio-inspired-navigation/system/controller/reachability_estimator/Snakefile, line 86:
Missing input files for rule 13:
    output: data/results/reachability_network-3colors-val.log
    wildcards: model=reachability_network-3colors, tag=
    affected files:
        data/models/reachability_network-3colors
```

2. Proper arguments for local controller
Please include one set of proper values for the arguments of local_controller in the README

## Comments (for Kejia herself)
### 1. Simulation Environment
- change to a top view.
- egocentric collision detection looks werid.
