rule:
	output: "system/controller/simulation/environment/{map}/maze_topview_binary.png"
	input: "system/controller/simulation/environment/{map}/plane.urdf"
	shell: "python scripts/create_occupancy_map.py {wildcards.map}"

maps = ['linear_sunburst', 'Savinov_val3']
pc_network_artifacts = ['env_coordinates', 'gc_connections', 'observations']

for mapname in maps:
	rule:
		output:
			[ f"system/bio_model/data/cognitive_map/{mapname}.after_exploration.gpickle" ] +
			[ f"system/bio_model/data/pc_model/{artifact}-{mapname}.npy" for artifact in pc_network_artifacts ]
		shell: f"python system/controller/topological/exploration_phase.py npc {mapname}"

for artifact in pc_network_artifacts:
	rule:
		input: f"system/bio_model/data/pc_model/{artifact}-Savinov_val3.npy"
		output: f"system/bio_model/data/pc_model/{artifact}.npy"
		shell: "ln -s $(basename {input}) {output}"
