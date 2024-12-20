set -e

experiments=( 
  '--spikings --lidar=raw_lidar'
  '--spikings --lidar=ego_bc'
  '--spikings --lidar=allo_bc'
  '--spikings'
  '--spikings --image-encoder fc --lidar=raw_lidar'
  '--lidar=raw_lidar'
  '--spikings --lidar=raw_lidar --dataset-features 3colors'
  '--spikings --lidar=raw_lidar --images=no'
  '--spikings --lidar=raw_lidar --dist'
  '--image-encoder fc --dist'
  '--image-encoder fc --lidar=ego_bc --dataset-features 3colors'
)

experiments_crop=(
  '--image-crop +10'
  '--image-crop -10'
  '--image-crop +20'
  '--image-crop -20'
  ''
)

for experiment in "${experiments[@]}"; do 
	tmux new-window -t training: -d "source thesis-quickstart.sh && for i in {1..5}; do nice python training/train_multiframe_dst.py train --tag \$i $experiment ; done || bash";
	# tmux new-session -d "source thesis-quickstart.sh && nice python training/train_multiframe_dst.py train --dropout $experiment || bash";
	# tmux new-session -d "source thesis-quickstart.sh && nice python training/train_multiframe_dst.py validate --dataset-basename dataset-val-2 $experiment; bash";
	# tmux new-session -d "source thesis-quickstart.sh && nice python training/train_multiframe_dst.py validate $experiment";
done
