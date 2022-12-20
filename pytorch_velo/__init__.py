import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from .optim import VeLO
