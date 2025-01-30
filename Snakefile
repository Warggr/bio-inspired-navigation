wildcard_constraints:
	thresh=r"0\.[0-9]+"

rule:
	output: "system/controller/simulation/environment/{map}/maze_topview_binary.png"
	input: "system/controller/simulation/environment/{map}/plane.urdf"
	shell: "python scripts/create_occupancy_map.py {wildcards.map}"

rule:
	output: [ f"system/controller/simulation/environment/{env}/maze_topview_binary.png" for env in ('Savinov_val3', 'final_layout') ]
	shell: "python scripts/create_occupancy_map_from_obj.py"

pc_network_artifacts = ['env_coordinates', 'gc_connections', 'observations']

rule:
	input: "system/controller/simulation/environment/{mapname}/maze_topview_binary.png"
	output:
		[ "system/bio_model/data/cognitive_map/{mapname}.after_exploration.gpickle" ] +
		[ f"system/bio_model/data/pc_model/{artifact}-{{mapname}}.npy" for artifact in pc_network_artifacts ]
	shell: "python system/controller/topological/exploration_phase.py -e {wildcards.mapname} npc"

for artifact in pc_network_artifacts:
	rule:
		input: f"system/bio_model/data/pc_model/{artifact}-Savinov_val3.npy"
		output: f"system/bio_model/data/pc_model/{artifact}.npy"
		shell: "ln -s $(basename {input}) {output}"

def maybe_require_nn(wildcards):
	if wildcards.re_type.startswith('neural_network'):
		nn_name = wildcards.re_type.removeprefix('neural_network(').removesuffix(')')
		return ["system/controller/reachability_estimator/data/models/{nn_name}"]
	return []

rule:
	output: "system/bio_model/data/cognitive_map/artifacts/{re_type}+threshold--{thresh}.gpickle"
	input: maybe_require_nn
	shell: """
		python system/controller/topological/exploration_phase.py --output-filename 'artifacts/{wildcards.re_type}+threshold--{wildcards.thresh}.gpickle' --re '{wildcards.re_type}' --mini threshold {wildcards.thresh}
	"""

rule:
	input: "system/bio_model/data/cognitive_map/artifacts/{re_type}+threshold--{tsame}.gpickle"
	output: "system/bio_model/data/cognitive_map/artifacts/{re_type}+threshold--{tsame}+treach--{treach}.gpickle"
	shell: "python scripts/cogmap_utils.py connect '{input}' '{wildcards.re_type}' '{output}' --threshold-reachable {wildcards.treach}"

rule:
	input: "system/bio_model/data/cognitive_map/artifacts/connect_re_mse_weights+threshold--{thresh}.gpickle"
	output: "logs/longnav_over_connect_cogmap_{thresh}.log"
	shell: """
		input={input}
		python system/tests/system_benchmark/long_unknown_nav.py Savinov_val3 ${{input#system/bio_model/data/cognitive_map/}} 0,-1 | tee {output}
	"""

rule:
	input: "system/bio_model/data/cognitive_map/artifacts/vo_{thresh}.gpickle"
	output: "logs/longnav_over_cogmap_{thresh}.log"
	shell: """
		input={input}
		python system/tests/system_benchmark/long_unknown_nav.py Savinov_val3 ${{input#system/bio_model/data/cognitive_map/}} 0,-1 | tee {output}
	"""

rule:
	output: "system/bio_model/data/bc_model/transformations.npz"
	shell: """ python system/bio_model/bc_network/bc_encoding.py """
