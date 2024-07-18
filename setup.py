from distutils.core import setup

setup(
    name='bio-inspired-navigation',
    version='0.1',
    packages=['system', 'system.plotting', 'system.controller', 'system.controller.simulation',
              'system.controller.local_controller', 'system.controller.local_controller.decoder',
              'system.controller.reachability_estimator', 'system.controller.reachability_estimator.training'],
    url='',
    license='',
    author='Pierre Ballif',
    author_email='pierre.ballif@tum.de',
    description=''
)
