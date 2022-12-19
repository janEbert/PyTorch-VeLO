import torch as th

from optim import VeLOOptimizer


def loss_with_backward(opt):
    p = next(iter(opt.param_groups))['params']
    opt.zero_grad()
    loss = th.mean(p[0]**2 + p[1]**2)
    loss.backward()
    return loss

devices = ['cpu']
if th.cuda.is_available():
    devices.append('cuda')

for device in devices:
    for dtype in [th.float32, th.float16]:
        init_params = [
            th.nn.Parameter(th.tensor([1.0], device=device)),
            th.nn.Parameter(th.tensor([1.0], device=device)),
        ]
        opt = VeLOOptimizer(init_params, num_training_steps=100)

        def closure():
            return loss_with_backward(opt)

        opt.step(closure)

        loss = loss_with_backward(opt)
        opt.step(closure)
