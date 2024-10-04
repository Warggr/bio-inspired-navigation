wildcard_constraints:
	thresh="0\.[0-9]+"

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

rule:
	output: "system/bio_model/data/cognitive_map/artifacts/nn({model})+threshold--{thresh}.gpickle"
	input: "system/controller/reachability_estimator/data/models/{model}.25"
	shell: """
		model={wildcards.model}
		python system/controller/topological/exploration_phase.py --output-filename 'artifacts/nn({wildcards.model})+threshold--{wildcards.thresh}.gpickle' --re 'neural_network({wildcards.model}.25)' --mini threshold {wildcards.thresh}
	"""

rule:
	input: "system/bio_model/data/cognitive_map/artifacts/connect_re_mse_weights+threshold--{thresh}.gpickle"
	output: "logs/longnav_over_connect_cogmap_{thresh}.log"
	shell: """
		input={input}
		python system/tests/system_benchmark/long_unknown_nav.py Savinov_val3 ${{input#system/bio_model/data/cognitive_map/}} 0,-1 | tee {output}
	"""

rule:
	input: "{map_in}.gpickle"
	output: "{map_in}+treach--{thresh}.gpickle"
	shell: "python scripts/cogmap_utils.py connect '{input}' 'neural_network(reachability_network-boolor+lidar--raw_lidar+conv.25)' '{output}' --threshold-reachable {wildcards.thresh}"

rule:
	input: "system/bio_model/data/cognitive_map/artifacts/vo_{thresh}.gpickle"
	output: "logs/longnav_over_cogmap_{thresh}.log"
	shell: """
		input={input}
		python system/tests/system_benchmark/long_unknown_nav.py Savinov_val3 ${{input#system/bio_model/data/cognitive_map/}} 0,-1 | tee {output}
	"""
