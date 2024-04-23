FEATURES = --spikings --lidar
FEATURES_FILE_FLAGS = +spikings+lidar

data/trajectories:
	mkdir -p $@

data/reachability:
	mkdir -p $@

data/trajectories/trajectories.hd5: data/trajectories
	python data_generation/gen_trajectories.py

data/reachability/dataset${FEATURES_FILE_FLAGS}.hd5: data/reachability
	python data_generation/dataset.py test -n 200000 --flush-freq 1000 --extension '.hd5.part' ${FEATURES}
	ln -s $(shell basename $@).part $@

data/reachability/trajectories.hd5: data/trajectories/trajectories.hd5
	ln -sf ../trajectories/trajectories.hd5 data/reachability/
