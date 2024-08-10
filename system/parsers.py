import argparse

controller_parser = argparse.ArgumentParser(add_help=False)
controller_parser.add_argument('--ray-length', default=1, type=float)
controller_parser.add_argument('--follow-walls', action='store_true')
controller_parser.add_argument('--tactile-cone', type=float, default=120)
