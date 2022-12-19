import inspect
from typing import Callable, Dict, Union

from learned_optimization.optimizers.base import (
    Optimizer as LearnedOptimizerBase,
)
from learned_optimization.research.general_lopt.prefab import (
    LearnedOptimizer,
)
from learned_optimization.research.general_lopt import pretrained_optimizers
import jax
import jax.numpy as jnp
import torch as th

LossClosure = Union[
    Callable[[], th.Tensor],
    Callable[[], float],
]

_DEFAULT_LOPT_FN = (
    inspect.signature(LearnedOptimizer).parameters['base_lopt_fn'].default
)
_TH_DTYPE_TO_JAX = {
    th.float16: jnp.float16,
    th.float32: jnp.float32,
}


def get_lopt_fn(opt_name: str, force=False) -> Callable:
    assert force or opt_name in pretrained_optimizers.opt_names, (
        'can only safely get pre-named optimizer functions. '
        'Supply `force=True` to ignore this error.'
    )
    lopt_name = opt_name.replace('.', '_').replace('-', '_')
    fn = getattr(pretrained_optimizers, lopt_name)
    assert callable(fn), f'{opt_name} does not resolve in a callable'
    return fn


def _th_dtype_to_jax(dtype: th.dtype) -> jnp.dtype:
    jax_dtype = _TH_DTYPE_TO_JAX.get(dtype)
    if jax_dtype is None:
        raise KeyError('unsupported dtype: ' + str(dtype))
    return jax_dtype


# FIXME convert dtypes
class VeLOOptimizer(th.optim.Optimizer):
    def __init__(
            self,
            params,
            num_training_steps: int,
            weight_decay: float = 0.0,
            max_training_steps: int = 150_000,
            base_lopt_fn: Callable[[], LearnedOptimizerBase] = (
                _DEFAULT_LOPT_FN
            ),
            seed: int = 0,
    ) -> None:
        defaults = dict(weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.opt = LearnedOptimizer(
            num_training_steps=num_training_steps,
            weight_decay=weight_decay,
            max_training_steps=max_training_steps,
            base_lopt_fn=base_lopt_fn,
        )

        jax_params = {
            str(i): [
                jnp.asarray(p.detach(), dtype=_th_dtype_to_jax(p.dtype))
                for p in group['params']
            ]
            for (i, group) in enumerate(self.param_groups)
        }

        rng_key = jax.random.PRNGKey(seed)
        self.state['rng_key'], init_key = jax.random.split(rng_key)
        self.state['opt_state'] = self.opt.init(
            jax_params,
            num_steps=num_training_steps,
            key=init_key,
        )

    @th.no_grad()
    def step(
            self,
            closure: LossClosure,
    ) -> Union[th.Tensor, float, None]:
        with th.enable_grad():
            closure_result = closure()
            if isinstance(closure_result, tuple):
                assert len(closure_result) == 2, (
                    'closure must return a 2-tuple if not returning a scalar'
                )
                loss, model_state = closure_result
            elif isinstance(closure_result, th.Tensor):
                loss = closure_result
                assert loss.numel() == 1, 'loss must be a scalar'
            else:
                raise TypeError('closure returned type that is not handled: ')

        jax_grad = {
            str(i): [
                jnp.asarray(p.grad, dtype=_th_dtype_to_jax(p.grad.dtype))
                for p in group['params']
            ]
            for (i, group) in enumerate(self.param_groups)
        }
        self.state['rng_key'], opt_key = jax.random.split(
            self.state['rng_key'])
        self.state['opt_state'] = self.opt.update(
            self.state['opt_state'],
            jax_grad,
            loss=jnp.asarray(loss, dtype=_th_dtype_to_jax(loss.dtype)),
            key=opt_key,
        )

        for (i, group) in enumerate(self.param_groups):
            for (param, jax_param) in zip(
                    group['params'],
                    self.opt.get_params(self.state['opt_state'])[str(i)],
            ):
                param.data = th.asarray(jax_param, dtype=param.data.dtype)
        return loss

    def __setstate__(self, state: Dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('weight_decay', 0.0)
