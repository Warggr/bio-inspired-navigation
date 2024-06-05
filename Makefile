data:
	mkdir data/

data/data.zip:
	wget https://syncandshare.lrz.de/dl/fi2PvsoTCgHNwra5QXrEwP/data.zip -O data/data.zip.part --continue && mv data/data.zip.part data/data.zip

system/bio_model/data/bc_model/transformations.npz:
	python system/bio_model/bc_network/bc_encoding.py

system/controller/reachability_estimator/data/models/re_mse_weights.50: data/data.zip
	unzip -p $< data/re/re_mse_weights.50 > $@

#system/controller/bio_model/data/cognitive_map/after_exploration.gpickle: data/data.zip
#	unzip -p $< data/bio_model/cognitive_map/after_exploration.gpickle > $@

system/bio_model/data/cognitive_map/after_exploration.gpickle:
	python system/controller/topological/exploration_phase.py
