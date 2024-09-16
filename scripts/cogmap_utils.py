from system.bio_model.cognitive_map import sample_normal

def add_connections_to_map(cognitive_map, re, add=True, remove=False):
    node_network = cognitive_map.node_network

    for i, ni in enumerate(node_network.nodes):
        for j, nj in enumerate(node_network.nodes):
            if i < j:
                continue
            if i != j:
                reachable, factor = re.get_reachability(ni, nj)
                if reachable and add and (nj not in node_network[ni] or 'connectivity_probability' not in node_network.edges[nj, ni]):
                    cognitive_map.add_bidirectional_edge_to_map(
                        ni, nj,
                        w=sample_normal(1-factor, cognitive_map.sigma),
                        connectivity_probability=re.get_connectivity_probability(factor),
                        mu=1-factor,
                        sigma=cognitive_map.sigma,
                    )
                if not reachable and remove and (nj in node_network[ni]):
                    cognitive_map.remove_bidirectional_edge(ni, nj)
    node_network.graph.update({'re_algorithm': str(re), 'threshold_reachable': re.threshold_reachable})

def guess_env_model(filename):
    match filename.split("."):
        case (env_model, _type, "gpickle"):
            return env_model
        case (_type, "gpickle"):
            return "Savinov_val3"
        case _:
            raise ValueError()
