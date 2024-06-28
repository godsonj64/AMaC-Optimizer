import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np


# Define the AMaC Optimizer class
class AMaC(optim.Optimizer):
    """
        Initialize the AMaC optimizer with the given parameters.
        
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            total_steps (int): Total number of training steps.
            lr (float, optional): Learning rate. Default: 0.001.
            beta1 (float, optional): Coefficient used for computing running averages of gradient. Default: 0.9.
            beta2 (float, optional): Coefficient used for computing running averages of squared gradient. Default: 0.999.
            mu (float, optional): Momentum factor. Default: 0.9.
            eps (float, optional): Term added to the denominator to improve numerical stability. Default: 1e-8.
            wd (float, optional): Weight decay (L2 penalty). Default: 0.01.
            alpha (float, optional): Coefficient used for Lookahead mechanism. Default: 0.5.
            k (int, optional): Number of steps before updating slow weights in Lookahead mechanism. Default: 5.
            clip_value (float, optional): Value for gradient clipping. Default: 1.0.
            warmup_steps (int, optional): Number of steps for learning rate warm-up. Default: 1000.
            swa_start (int, optional): Step to start Stochastic Weight Averaging (SWA). Default: 10.
            noise_factor (float, optional): Factor for adaptive noise injection. Default: 0.01.
        """
    def __init__(self, params, total_steps, lr=0.001, beta1=0.9, beta2=0.999, mu=0.9, eps=1e-8, wd=0.01, alpha=0.5, k=5,
                 clip_value=1.0, warmup_steps=1000, swa_start=10, noise_factor=0.01):
        self.total_steps = total_steps  # Store total_steps as an instance variable
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, mu=mu, eps=eps, wd=wd, alpha=alpha, k=k, clip_value=clip_value,
                        warmup_steps=warmup_steps, swa_start=swa_start, noise_factor=noise_factor)  # Define the default parameter settings for the optimizer
        super(AMaC, self).__init__(params, defaults)   # Initialize the optimizer with the given parameters and defaults
        self.swa_weights = []# List to store SWA (Stochastic Weight Averaging) weights
        self.swa_start = swa_start# Step at which to start SWA
        self.swa_n = 0 # Counter for the number of SWA updates

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['u'] = torch.zeros_like(p.data)
                    state['slow_weights'] = torch.clone(p.data).detach()
                    state['warmup_lr'] = group['lr'] / group['warmup_steps']
                m, v, u, slow_weights = state['m'], state['v'], state['u'], state['slow_weights']
                beta1, beta2, mu, eps, wd, alpha, k, clip_value, warmup_steps, noise_factor = group['beta1'], group[
                    'beta2'], group['mu'], group['eps'], group['wd'], group['alpha'], group['k'], group['clip_value'], \
                group['warmup_steps'], group['noise_factor']
                state['step'] += 1
                t = state['step']
                if t <= warmup_steps:
                    lr = state['warmup_lr'] * t
                else:
                    lr = group['lr'] * (
                                0.5 * (1 + math.cos(math.pi * (t - warmup_steps) / (self.total_steps - warmup_steps))))
                grad -= grad.mean()
                m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                delta = m_hat / (v_hat.sqrt() + eps)
                noise_level = noise_factor * (1 - t / (2 * warmup_steps))
                noise = noise_level * torch.randn_like(delta)
                delta.add_(noise)
                delta = torch.clamp(delta, -clip_value, clip_value)
                weight_decay = wd * (1 - t / (2 * warmup_steps))
                p.data.mul_(1 - lr * weight_decay)
                u.mul_(mu).add_(delta, alpha=-lr)
                p.data.add_(u)
                if t % k == 0:
                    slow_weights.mul_(alpha).add_(p.data, alpha=(1 - alpha))
                    p.data.copy_(slow_weights)
                if t >= warmup_steps and t % k == 0:
                    if len(self.swa_weights) == 0:
                        self.swa_weights = [torch.zeros_like(p.data) for p in group['params']]
                    for swa_w, p in zip(self.swa_weights, group['params']):
                        swa_w.mul_(self.swa_n / (self.swa_n + 1)).add_(p.data / (self.swa_n + 1))
                    self.swa_n += 1
        return loss

    def swap_swa_weights(self):
        for group in self.param_groups:
            for swa_w, p in zip(self.swa_weights, group['params']):
                p.data.copy_(swa_w)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

