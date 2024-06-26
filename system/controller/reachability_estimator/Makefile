FEATURES = --spikings --lidar=allo_bc
FEATURES_FILE_FLAGS = +spikings+lidar

folders = data/trajectories data/reachability data/models data/results

$(folders): %:
	mkdir -p $@

data/trajectories/trajectories.hd5: | data/trajectories
	python data_generation/gen_trajectories.py --extension '.hd5.part'
	ln -s $(shell basename $@).part $@

data/reachability/trajectories.hd5: data/trajectories/trajectories.hd5 | data/reachability
	ln -sf ../trajectories/trajectories.hd5 $@

data/reachability/dataset.hd5: data/reachability/trajectories.hd5 | data/reachability
	python data_generation/dataset.py -n 200000 --flush-freq 1000 --extension '.hd5.part'
	ln -s $(shell basename $@).part $@

data/reachability/dataset-3colors.hd5: | data/reachability
	python data_generation/dataset.py -n 200000 --flush-freq 1000 --extension '.hd5.part' -wall-colors 3colors
	ln -s $(shell basename $@).part $@

data/models/reachability_network${FEATURES_FILE_FLAGS}.hd5: data/reachability/dataset.hd5 | data/models
	python training/train_multiframe_dst.py train ${FEATURES} --resume

data/models/reachability_network-3colors${FEATURES_FILE_FLAGS}.hd5: data/reachability/dataset-3colors.hd5 | data/models
	python training/train_multiframe_dst.py train ${FEATURES} --resume --dataset-features 3colors

code_sizes = 0 1 4 16 20 100

AUTOENCODER_EPOCHS = 25

autoencoders := $(addsuffix .${AUTOENCODER_EPOCHS},$(addprefix data/models/autoencoder,${code_sizes}))

$(autoencoders): data/models/autoencoder%.${AUTOENCODER_EPOCHS}: data/reachability/dataset.hd5
	python training/train_autoencoder.py train $* -e ${AUTOENCODER_EPOCHS}

autoencoders_performance := $(subst models,results,$(addsuffix .log, ${autoencoders}))
$(autoencoders_performance): data/results/autoencoder%.${AUTOENCODER_EPOCHS}.log: data/models/autoencoder%.${AUTOENCODER_EPOCHS} | data/results
	python training/train_autoencoder.py validate $* > $@

data/results/autoencoder_performance.csv: $(autoencoders_performance) | data/results
	{ echo "Dimension of the encoding,MSE loss" &&\
	  for i in $(code_sizes); do\
		echo -n $$i, &&\
		cat data/results/autoencoder$$i.${AUTOENCODER_EPOCHS}.log | cut -f 2 \
	; done } > $@

data/results/autoencoder_performance.png: data/results/autoencoder_performance.csv
	python scripts/plot.py $< $@ "Autoencoder performance"
