# coding: utf-8
import numpy as np
from system.controller.local_controller.local_navigation import create_gc_spiking, setup_gc_network, PodGcCompass

gc_network = setup_gc_network(dt=1e-2)
goal_spiking = create_gc_spiking(start=(0, 0), goal=(5, 0))#, gc_network_at_start=gc_network)
compass = PodGcCompass(pod_network=None, gc_network=gc_network)

#compass.reset_goal(gc_network.consolidate_gc_spiking())
gc_network.set_current_as_target_state()
assert np.linalg.norm(compass.calculate_goal_vector()) <= 0.1

#compass.reset_goal(goal_spiking)
gc_network.set_as_target_state(goal_spiking)
assert np.linalg.norm(compass.calculate_goal_vector()) >= 4, compass.calculate_goal_vector()
