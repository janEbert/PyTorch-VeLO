from setuptools import setup

setup(
    name='pytorch_velo',
    python_requires='>=3.8',
    version='0.0.1',
    install_requires=[
        (
            'learned_optimization '
            '@ git+https://github.com/google/learned_optimization.git'
        ),
        'optax>=0.1',
        'torch',
    ]
)
