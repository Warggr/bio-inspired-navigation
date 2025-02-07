# This Makefile creates files by downloading the values already generated by Anna and uploaded on SyncAndShare.
# To create the files yourself, use system/controller/reachability_estimator/Makefile

rule:
	output: "data/data.zip"
	shell: """
		wget https://syncandshare.lrz.de/dl/fi2PvsoTCgHNwra5QXrEwP/data.zip -O data/data.zip.part --continue
		mv data/data.zip.part data/data.zip
	"""

pc_network_artifacts = ['env_coordinates', 'gc_connections', 'observations']
pc_network_artifact_files = [ f"system/bio_model/data/pc_model/{artifact}.npy" for artifact in pc_network_artifacts ]

rule pc_artifact:
	input: "data/data.zip"
	output: 'system/bio_model/data/pc_model/{artifact}.npy'
	shell: "unzip -p {input} data/bio_model/place_cells/{wildcards.artifact}.npy > {output}"

rule pc_network:
	input: pc_network_artifact_files

gc_network_artifacts = ['gm_values', 'h_vectors', 's_vectors_initialized', 'w_vectors']
gc_network_artifact_files = [ f"system/bio_model/data/gc_model_6/{artifact}.npy" for artifact in gc_network_artifacts ]

rule gc_artifact:
	input: "data/data.zip"
	output: 'system/bio_model/data/gc_model_6/{artifact}.npy'
	shell: "unzip -p {input} data/bio_model/grid_cells/{wildcards.artifact}.npy > {output}"

rule gc_network:
	input: gc_network_artifact_files

rule:
	output: "data/data_pierre.zip"
	shell: """
		wget 'https://syncandshare.lrz.de/dl/fiPMNYz94gDzS7tZyETPuT/data_pierre.zip' -O {output}.part --continue
		mv {output}.part {output}
	"""

artifacts_pierre = [
	'system/controller/reachability_estimator/data/models/reachability_network+spikings+lidar--raw_lidar+conv.25',
	'system/controller/reachability_estimator/data/models/reachability_network-boolor+lidar--raw_lidar+conv.25',
	'system/bio_model/data/bc_model/transformations.npz',
]

for artifact in artifacts_pierre:
	rule:
		input: "data/data_pierre.zip"
		output: artifact
		shell: """ unzip {input} {output}"""
