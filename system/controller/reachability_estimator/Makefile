FEATURES = --spikings --lidar=allo_bc
FEATURES_FILE_FLAGS = +spikings+lidar

data/trajectories:
	mkdir -p $@

data/reachability:
	mkdir -p $@

data/models:
	mkdir -p $@

data/trajectories/trajectories.hd5: data/trajectories
	python data_generation/gen_trajectories.py

data/reachability/dataset.hd5: data/reachability ../../bio_model/data/bc_model/transformations.npz
	python data_generation/dataset.py test -n 200000 --flush-freq 1000 --extension '.hd5.part' ${FEATURES}
	ln -s $(shell basename $@).part $@

data/reachability/trajectories.hd5: data/trajectories/trajectories.hd5
	ln -sf ../trajectories/trajectories.hd5 data/reachability/

data/models/reachability_network${FEATURES_FILE_FLAGS}.hd5: data/reachability/dataset.hd5
	python training/train_multiframe_dst.py train ${FEATURES} --resume
