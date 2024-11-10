import math
from contextlib import ExitStack
from typing import Optional

from system.controller.local_controller.compass import AnalyticalCompass, Compass
from system.controller.local_controller.local_controller import LocalController, controller_rules
from system.controller.local_controller.local_navigation import vector_navigation
from system.controller.simulation.environment_cache import EnvironmentCache
from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.types import AllowedMapName, Angle, Vector2D
from system.debug import PLOTTING

def test(
    controller: LocalController,
    env_name: AllowedMapName,
    start: tuple[Vector2D, Angle],
    goal: tuple[Vector2D, Angle],
    compass: Optional[Compass] = None,
    env: Optional[PybulletEnvironment] = None,
    robot: Optional[Robot] = None,
    visualize: bool = False,
) -> tuple[bool, int]:
    with ExitStack() as context:
        if env is None:
            env = PybulletEnvironment(env_name, visualize=visualize, contains_robot=False)
            context.push(env)
        else:
            assert env.env_model == env_name
            robot = env.robot
        if robot is None:
            robot = Robot(env, base_position=start[0], base_orientation=start[1])
            robot.__enter__()
            context.push(robot)
        else:
            assert robot.position_and_angle == start
        if compass is None:
            compass = AnalyticalCompass(start_pos=start[0], goal_pos=goal[0])
        reached, nr_steps = vector_navigation(env, compass, controller=controller, step_limit=2000, collect_nr_steps=True)
    return reached, nr_steps

if __name__ == "__main__":
    import sys
    import argparse
    from system.parsers import controller_parser, controller_creator

    parser = argparse.ArgumentParser(parents=(controller_parser,))
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('experiment', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if len(args.experiment) == 0:
        cases = [line.strip().split(" ") for line in sys.stdin]
    elif len(args.experiment) == 3:
        cases = [args.experiment]
    else:
        parser.error(f'Usage:\n\tpython {sys.argv[0]} [OPTIONS]\nor\n\tpython {sys.argv[0]} [OPTIONS] map_name start goal')

    total_reached, total_steps = 0, 0
    with EnvironmentCache(override_env_kwargs={'visualize': args.visualize}) as cache:
        for map_name, start, goal in cases:
            start, goal = map(lambda csv: tuple(map(float, csv.split(','))), (start, goal))
            start, goal = map(lambda tup: ((tup[0], tup[1]), math.radians(tup[2])), (start, goal))
            controller = controller_creator(args)
            reached, nr_steps = test(controller, map_name, start, goal, env=cache[map_name], visualize=args.visualize)
            print(reached, nr_steps)
            total_reached += reached; total_steps += nr_steps
    if len(cases) >= 2:
        print(f'{len(cases)} tests summary:'
              f' {total_reached} ({total_reached / len(cases)}) reached,'
              f' steps: {total_steps} total, {total_steps / len(cases)} avg')
