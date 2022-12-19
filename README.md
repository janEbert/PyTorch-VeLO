# PyTorch-VeLO

[VeLO optimizer](https://arxiv.org/abs/2211.09760) usable from
PyTorch.

The wrapping is very basic, we try to let JAX do everything so we do
not have to re-implement the optimizer in PyTorch.

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is automatically set so JAX does
not consume all GPU memory.

## Caution

Alpha-level software. Not well tested, probably highly imperformant.

Currently parameters are **completely copied** if using a GPU.

Need to figure out if/how often tensors are copied.
