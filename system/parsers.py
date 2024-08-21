import argparse

controller_parser = argparse.ArgumentParser(add_help=False)
controller_parser.add_argument('--ray-length', default=1, type=float)
controller_parser.add_argument('--follow-walls', action='store_true')
controller_parser.add_argument('--tactile-cone', type=float, default=120)
controller_parser.add_argument('--position-estimation', action='store_true')

def controller_creator(args):
    from system.controller.local_controller.local_controller import LocalController, controller_rules
    import numpy as np
    controller = LocalController(
        on_reset_goal=[controller_rules.TurnToGoal()],
        transform_goal_vector=[controller_rules.ObstacleAvoidance(ray_length=args.ray_length, tactile_cone=np.radians(args.tactile_cone), follow_walls=args.follow_walls)],
        hooks=[controller_rules.StuckDetector(400)],
    )
    return controller
