import system.plotting.plotResults as plot
from system.plotting.plotHelper import add_environment
from system.bio_model.cognitive_map import LifelongCognitiveMap
import matplotlib.pyplot as plt


def draw_map(file: str, env_model, env_variant=None, cognitive_map: LifelongCognitiveMap|None=None, *, show=False, **kwargs):
    default_variants = {'linear_sunburst': 'plane'}
    env_variant = env_variant or default_variants.get(env_model, None)
    if cognitive_map is None:
        cognitive_map = LifelongCognitiveMap(reachability_estimator=None, load_data_from=file, debug=False)
    print(f"{file} ({len(cognitive_map.node_network.nodes)} nodes)")
    print("metadata:", cognitive_map.node_network.graph)
    assert file.endswith('.gpickle'); filename_stripped = file.removesuffix('.gpickle')
    for metric in ['coverage', 'mean-distance', 'edges']:
        try:
            with open(f'../system/bio_model/data/cognitive_map/results/{filename_stripped}-{metric}.v') as file:
                value = float(file.read())
            print(metric, ':', value)
        except FileNotFoundError:
            pass
    fig, ax = plt.subplots()
    add_environment(ax, env_model, variant=env_variant)
    plot.plotCognitiveMap(ax, cognitive_map, **kwargs)
    if show:
        plt.show()
