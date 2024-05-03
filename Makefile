data:
	mkdir data/

data/data.zip:
	wget https://syncandshare.lrz.de/dl/fi2PvsoTCgHNwra5QXrEwP/data.zip -O data/data.zip.part --continue && mv data/data.zip.part data/data.zip

system/bio_model/data/bc_model/transformations.npz:
	python system/bio_model/bc_network/bc_encoding.py
