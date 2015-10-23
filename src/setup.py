from distutils.core import setup

setup(
    name='crowdDynamics',
    version='1.7',
    packages=['numpy', 'matplotlib', 'cvxopt', 'networkx', 'scipy'],
    package_dir={'': 'src'},
    url='https://github.com/0mar/crowd_simulation',
    license='MIT',
    author='0mar',
    author_email='omsrichardson@gmail.com',
    description='A novel implementation for a hybrid crowd dynamics model'
)
