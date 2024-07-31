rule:
	output: "data/data.zip"
	shell: """
	wget https://syncandshare.lrz.de/dl/fi2PvsoTCgHNwra5QXrEwP/data.zip -O data/data.zip.part --continue
	mv data/data.zip.part data/data.zip
	"""



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
		shell: f"python system/controller/topological/exploration_phase.py {mapname}"

for artifact in pc_network_artifacts:
	rule:
		input: f"system/bio_model/data/pc_model/{artifact}-Savinov_val3.npy"
		output: f"system/bio_model/data/pc_model/{artifact}.npy"
		shell: "ln -s $(basename {input}) {output}"

rule:
	output: "system/tests/results/local_controller_angle/length={raylen}.log"
	shell: """
		echo -n '{wildcards.raylen}:' > {output}
		python system/tests/local_controller_angle_test.py --ray-length {wildcards.raylen} | tail -n 1 >> {output}
	"""

rule:
	input: [ f"system/tests/results/local_controller_angle/length={raylen}.log" for raylen in [0, 0.1, 0.2, 0.5, 0.8, 1, 1.5, 2] ]
	output:
		csv = "system/tests/results/local_controller_angle/results.csv",
		pgf = "system/tests/results/local_controller_angle/results.pgf",
		png = "system/tests/results/local_controller_angle/results.png",
		#{format=f"system/tests/results/local_controller_angle/results.{format}" for format in ['pgf', 'png', 'csv']}
	run:
		results = []
		for line in shell(f"cat {input}", iterable=True):
			raylen, line = line.split(':')
			raylen = float(raylen)
			LINE_START = 'Maximum handle-able angle is '
			assert line.startswith(LINE_START)
			line = line.removeprefix(LINE_START)
			start, stop = line.split(' ~ ')
			start, stop = [int(i) for i in (start, stop)]
			results.append( {{ 'start': start, 'stop': stop, 'raylen': raylen }})
		import pandas as pd
		dset = pd.DataFrame(results)
		dset.to_csv('system/tests/results/local_controller_angle/result.csv')
		import matplotlib
		import matplotlib.pyplot as plt
		matplotlib.use("pgf")
		plt.scatter(dset['raylen'], dset['start'], label='lower bound')
		plt.scatter(dset['raylen'], dset['stop'], label='upper bound')
		plt.xlabel('Ray length')
		plt.ylabel('Maximum handled angle')
		plt.legend()
		plt.savefig(output.png)
		plt.savefig(output.pgf)
