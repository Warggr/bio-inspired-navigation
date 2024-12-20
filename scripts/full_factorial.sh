#!/usr/bin/bash

for SAMPLES_PER_EPOCH in 1000 10000 20000 ; do for LR in 0.003 0.0003 0.00005 ; do for _ in {1..3}; do tmux new-session -d "source thesis-quickstart.sh && nice python training/train_multiframe_dst.py train --dataset-features 3colors --spikings --lidar=raw_lidar --pair-conv --samples-per-epoch=$SAMPLES_PER_EPOCH --lr=$LR; bash" ; done ; done ; done
