# PyTorch-VeLO

[VeLO optimizer](https://arxiv.org/abs/2211.09760) usable from
PyTorch.

The wrapping is very basic, we try to let JAX do everything so we do
not have to re-implement the optimizer in PyTorch.

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is automatically set so JAX does
not consume all GPU memory.

## Installation

```shell
python3 -m pip install git+https://github.com/janEbert/PyTorch-VeLO.git
```

## Usage

```python
from pytorch_velo import VeLO

[...]

train_steps = epochs * len(dataset)  # Assuming `dataset` is already batched.
opt = VeLO(params, num_training_steps=train_steps, weight_decay=0.0)

# Use like any other PyTorch optimizer.
```

## Caution

Alpha-level software. Not well tested, probably highly imperformant.

Only parameters with trivial strides are supported; this will have to
be implemented on the JAX side (see
https://github.com/google/jax/issues/8082).

With `jax==0.3.21` (automatically installed via `learned_optimization`
as of writing), the `jax.default_device` context manager does not
work. To force JAX to use the CPU for its optimizer, set the
environment variable `JAX_PLATFORMS=cpu`.
