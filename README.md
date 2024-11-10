# BA - Bio-Inspired Navigation

Welcome to the Git Repo for the Master Thesis on

> Integrating Navigation Strategies and Dynamic Cognitive Mapping 
> for Biologically Inspired Navigation

In this file we explain the installation process and describe the use of some of the different files.
For a detailed description and the reasoning behind this code, please refer to the thesis.


## Install packages
The code is based on Python3. It has been tested on Python 3.11 and 3.12 and might not work on other versions.

Install the following packages to run the code:

Install the required packages with
```shell
        pip install -r requirements.txt
```
A gcc,g++ and latex installation are required and can be added with the following commands if not installed already. 
```shell
        sudo apt install gcc
        sudo apt install g++

        sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
        pip install latex
```
You also need to setup the modified version of range-libc, that can be found in this repository. The original version can be found at https://github.com/kctess5/range_libc.
It had to be modified to work with Python3. Follow these instructions (https://github.com/kctess5/range_libc#python-wrappers) to install.
```shell
        cd range_libc/pywrapper
        pip install .
        python test.py
```

## More Setup

Precomputed weights for reachability estimator, place and grid cells, 
as well as multiple versions of cognitive maps can be found under 
[this link](https://syncandshare.lrz.de/getlink/fi2PvsoTCgHNwra5QXrEwP/data.zip).

To download the weights and put every file in the expected folder, you can use Make (see [here](./Makefile)) or the more modern Snakemake (see [here](./Snakefile)).

This folder contains:
- bio_data: model of grid cells, place cells and the cognitive map
- re: trajectory and reachability dataset for training as well as the final model

## Debugging

The code reads the environment variables `DEBUG` and `PLOTTING`.
Depending on their contents, multiple debug configurations can be activated.
To debug multiple things, you can separate them by the sign `&`, e.g.
```shell
export DEBUG='localctrl&gains'
```

#### Flags for DEBUG
- `gains`: prints a line with the robot gains at each simulation time step.
- `cogmap`: prints whenever the cognitive map is modified

#### Flags for PLOTTING
- `exploration`: plots trajectories used for exploration phase

## Known bugs and corresponding fixes
### If you encounter bugs related to OpenGL
(@TUM students: this happens when running on LXHalle computers)

This can fix the bug in some cases:
```shell
export LIBGL_DRIVERS_PATH=/lib/x86_64-linux-gnu/dri
export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6.0.30
```

### Interactions between Matplotlib and Bullet
When Matplotlib plots are plotted before the simulation runs, really bizarre errors can happen, e.g. walls are horizontal instead of vertical, the car rolls sideways instead of horizontally on its wheels, and/or the car passes through the floor.
See e.g. [here](https://github.com/bulletphysics/bullet3/issues/2125).

A workaround is to create a shared memory server:
```shell
# run Bullet server in the background
python -m pybullet_utils.runServer &
# export environment variable to tell our code to use the server
export BULLET_SHMEM_SERVER=True
```

You can reuse the same server for multiple script runs - each script cleans up the simulation environment after it has run.

## Code
This code implements the methodology as well as performs the experiments described in the thesis.

### Simulation
[system/controller/simulation/pybullet_environment.py](system/controller/simulation/pybullet_environment.py)

Test different environment and the camera by moving an agent with your keyboard and plotting its trajectory. Change between four different environments.
Press arrow keys to move, SPACE to visualize egocentric rays with obstacle detection and  BACKSPACE to exit.

Available environments:

    - plane
    - obstacle_map_0    --\
    - obstacle_map_1       \ Environments to test obstacle avoidance
    - obstacle_map_2       /
    - obstacle_map_3    --/
    - Savinov_test7
    - Savinov_val2
    - Savinov_val3 (default for all project stages)


----

### Local Controller
[system/controller/local_controller/local_navigation.py](system/controller/local_controller/local_navigation.py)

Test navigating with the local controller using different movement vector and goal vector calculation methods in different environments.

Available decoders:
- pod: phase-offset decoder
- linear_lookahead: linear lookahead decoder
- analytical: precise calculation of goal vector with information that is not biologically available to the agent
- combo: uses pod until < 0.5 m from the goal, then switches to linear_lookahead for higher precision 
        The navigation uses the faster pod decoder until the agent thinks it has reached its goal, 
        then switches to slower linear lookahead for increased accuracy.

Calculation of movement vector:
- obstacles = True: enable obstacle avoidance to create the movement vector
- obstacles = False: the movement vector is the goal vector

#### Obstacle Avoidance Test
[system/controller/local_controller/local_navigation.py](system/controller/local_controller/local_navigation.py)

Perform the experiments described in subsection 5.5.3 Obstacle Avoidance.
Set ***experiment = "obstacle_avoidance"***

- Choose between decoders: ***model = "analytical" or "combo"***
- Adjust which parameters to test
    - ***all = True*** : test entire range of parameter combinations
    - ***all = False*** : manually choose which combinations to test

To plot any of the attempts set ***plot_it = True***.

----

### Reachability Estimator
[system/controller/reachability_estimator/reachability_estimation.py](system/controller/reachability_estimator/reachability_estimation.py)

There are several methods of judging reachability available:
- ***type = "distance"***: return distance between nodes
- ***type = "neural_network"***: use the neural model
- ***type = "simulation"***: simulate the navigation and return success or failure (only works as a connection RE)
- ***type = "view_overlap"***: return the view overlap between the nodes (only works as a connection RE)

To adjust what values are considered as reachable adjust the creation and connection thresholds in pc_network.py and cognitivemap.py.

#### Trajectory Generation
[system/controller/reachability_estimator/data_generation/gen_trajectories.py](system/controller/reachability_estimator/data_generation/gen_trajectories.py)

Generate trajectories through the environment storing grid cell spikings and coordinates.

Testing:
Generate/ load a few trajectories per map and display.

Default:
Generate 1000 trajectories of length 3000 with a saving frequency of 10 
in the environment "Savinov_val3"

Parameterized:
Adjust filename, env_model, num_traj, traj_length and cam_freq 

#### Reachability Dataset Generation
[system/controller/reachability_estimator/data_generation/dataset.py](system/controller/reachability_estimator/data_generation/dataset.py)

Generate a dataset of reachability samples or load from an existing one.
    
Testing:
Generate or load and display a few samples.

Parameterized call:
- Save to/load from filename
- Generate until there are num_samples samples
- Use trajectories from traj_file
    
Default: Time the generation of 50 samples

----

### Topological Navigation

#### Exploration
[system/controller/topological/exploration_phase.py](system/controller/topological/exploration_phase.py)

Perform the experiments described in subsection 5.3 Cognitive Map Construction

Create a cognitive map by exploring the environment.
Adjust ***connection_re_type*** and ***creation_re_type***:
-  types: "firing", "neural_network", "distance", "simulation", "view_overlap"
    - "firing": place cell firing value
    - others: see explanation for RE

To adjust what values are considered as reachable adjust the creation and connection thresholds in pc_network.py and cognitivemap.py.

#### Cognitive Map
[system/bio_model/cognitive_map.py](system/bio_model/cognitive_map.py)

Perform the experiments described in subsection 6.3 Cognitive Map Construction

Update the connections on the cognitive map or draw it.

#### Navigation
[system/controller/topological/topological_navigation.py](system/controller/topological/topological_navigation.py)

Perform the experiments described in subsection 6.4.1 Topological Navigation and 6.6 Overall Performance
Test navigation through the maze.

----

#### Plotting
[system/plotting/*.py](https://github.com/Fedannie/bio-inspired-navigation/tree/main/system/plotting)

Functions that don't have direct references from the code were used 
by generations of students for plot generation for the thesis.
They're not relevant for the project and might not work as expected.
However, if you need to create a plot similar to the one in someone else's thesis,
take a look at the functions in the underlying files, there might be something useful already.


## Code Structure

        .
        ├── range_libc                                  # Modified version of range_libc
        ├── system                                      # Navigation system
        │   ├── bio_model                               # Scripts modeling biological entities
        │   │   ├── place_cell_model.py                 # Implements place cells
        │   │   ├── grid_cell_model.py                  # Implements grid cells
        │   │   ├── cognitive_map.py                    # Implements a cognitive map and lifelong learning
        │   ├── controller                              # Scripts controlling the agent and environment
        │   │   ├── local_controller                    # Performs local navigation
        │   │   │   ├── decoder                         # Scripts for different grid cell decoder mechanism
        │   │   │   └── local_navigation.py             # Implements vector navigation and obstacle avoidance
        │   │   ├── reachability_estimator              # Scripts and data for reachability estimation
        │   │   │   ├── data                            # Stores generated trajectories and reachability dataset
        │   │   │   ├── data_generation                 # Scripts for trajectory and reachability dataset generation
        │   │   │   ├── training                        # Scripts for training the reachability estimator and training data
        │   │   │   ├── networks.py                     # Separate neural network modules that participate in the reachability estimator structure
        │   │   │   └── reachability_estimation.py      # Implements different kinds of reachability estimation
        │   │   ├── simulation                          # Scripts and data for simulating the agent in the environment
        │   │   │   ├── environment                     # Scripts and data concerning the environment
        │   │   │   ├── p3dx                            # Data for the agent
        │   │   │   └── pybullet_environment.py         # Implements simulation steps in the environment
        │   │   └── topological                         # Scripts for topological navigation
        │   │   │   ├── exploration_phase.py            # Implements exploration of the environment
        │   │   │   └── topological_navigation.py       # Implements topological navigation in the environment
        │   ├── plotting                                # Scripts to create supplementary plots
        ├── README.md                                   # You are here. Overview of project
        └── requirements.txt                            # Required packages to install
## Further Questions

To ask questions about the thesis or code please reach out to anna.fedorova.se@gmail.com

## Acknowledgement

This code was largely inspired and adapted from Latzel(2023)[^1], Engelmann(2021)[^2] and Meng(2022)[^3]


[^1]: Latzel Johanna, "Neurobiologically inspired Navigation for Artificial Agents", Sept. 2023
[^2]: Engelmann Tim, "Biologically inspired spatial navigation using vector-based
and topology-based path planning", Sept. 2021
[^3]: Meng, X., N. Ratliff, Y. Xiang, and D. Fox , "Scaling Local Control to Large-
Scale Topological Navigation." 2020
