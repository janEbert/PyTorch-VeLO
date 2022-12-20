# PyTorch-VeLO

[VeLO optimizer](https://arxiv.org/abs/2211.09760) usable from
PyTorch.

The wrapping is very basic, we try to let JAX do everything so we do
not have to re-implement the optimizer in PyTorch.

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is automatically set so JAX does
not consume all GPU memory.

## Caution

Alpha-level software. Not well tested, probably highly imperformant.

With `jax==0.3.21` (automatically installed via `learned_optimization`
as of writing), the `jax.default_device` context manager does not
work. To force JAX to use the CPU for its optimizer, set the
environment variable `JAX_PLATFORMS=cpu`.
