import math
import torch
from torch.optim import Optimizer

class AMaC(Optimizer):
    """
    AMaC Optimizer - Adaptive Momentum and Cosine Annealing Optimizer
    By: Godson Johnson

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
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

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, mu=0.9, eps=1e-8, wd=0.01, alpha=0.5, k=5, clip_value=1.0, warmup_steps=1000, swa_start=10, noise_factor=0.01):
        # Define default parameters for the optimizer
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, mu=mu, eps=eps, wd=wd, alpha=alpha, k=k, clip_value=clip_value, warmup_steps=warmup_steps, swa_start=swa_start, noise_factor=noise_factor)
        # Initialize the base optimizer class with the parameters
        super(AMaC, self).__init__(params, defaults)
        
        # Initialize variables for Stochastic Weight Averaging (SWA)
        self.swa_weights = []  # List to store SWA weights
        self.swa_start = swa_start  # Step to start SWA
        self.swa_n = 0  # Counter for SWA steps

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Iterate over each parameter group
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  # Initialize first moment vector
                    state['v'] = torch.zeros_like(p.data)  # Initialize second moment vector
                    state['u'] = torch.zeros_like(p.data)  # Initialize momentum update vector
                    state['slow_weights'] = torch.clone(p.data).detach()  # Initialize slow weights for Lookahead
                    state['warmup_lr'] = group['lr'] / group['warmup_steps']  # Compute warmup learning rate

                # Unpack state variables
                m, v, u, slow_weights = state['m'], state['v'], state['u'], state['slow_weights']
                beta1, beta2, mu, eps, wd, alpha, k, clip_value, warmup_steps, noise_factor = group['beta1'], group['beta2'], group['mu'], group['eps'], group['wd'], group['alpha'], group['k'], group['clip_value'], group['warmup_steps'], group['noise_factor']

                state['step'] += 1
                t = state['step']

                # Learning rate warm-up and cosine annealing
                if t <= warmup_steps:
                    lr = state['warmup_lr'] * t  # Linearly increase learning rate during warm-up
                else:
                    # Apply cosine annealing after warm-up period
                    lr = group['lr'] * (0.5 * (1 + math.cos(math.pi * (t - warmup_steps) / (warmup_steps * 2))))

                # Gradient Centralization: subtract mean of gradients to center them
                grad -= grad.mean()

                # Update biased first moment estimate (moving average of the gradients)
                m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                # Update biased second raw moment estimate (moving average of the squared gradients)
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - beta1 ** t)
                # Compute bias-corrected second raw moment estimate
                v_hat = v / (1 - beta2 ** t)

                # Compute update step
                delta = m_hat / (v_hat.sqrt() + eps)

                # Adaptive Noise Injection: inject noise into the update
                noise_level = noise_factor * (1 - t / (warmup_steps * 2))
                noise = noise_level * torch.randn_like(delta)
                delta.add_(noise)

                # Gradient Clipping: clip the gradients to prevent explosion
                delta = torch.clamp(delta, -clip_value, clip_value)

                # Dynamic Weight Decay: adjust weight decay dynamically
                weight_decay = wd * (1 - t / (warmup_steps * 2))
                p.data.mul_(1 - lr * weight_decay)

                # Momentum update
                u.mul_(mu).add_(delta, alpha=-lr)
                p.data.add_(u)

                # Lookahead mechanism: periodically update slow weights
                if t % k == 0:
                    slow_weights.mul_(alpha).add_(p.data, alpha=(1 - alpha))
                    p.data.copy_(slow_weights)

                # Stochastic Weight Averaging (SWA) with Exponential Moving Average
                if t >= warmup_steps and t % k == 0:
                    if len(self.swa_weights) == 0:
                        # Initialize SWA weights if not already done
                        self.swa_weights = [torch.zeros_like(p.data) for p in group['params']]
                    for swa_w, p in zip(self.swa_weights, group['params']):
                        # Update SWA weights using exponential moving average
                        swa_w.mul_(self.swa_n / (self.swa_n + 1)).add_(p.data / (self.swa_n + 1))
                    self.swa_n += 1

        return loss

    def swap_swa_weights(self):
        """
        Swap current weights with SWA (Stochastic Weight Averaging) weights.
        """
        for group in self.param_groups:
            for swa_w, p in zip(self.swa_weights, group['params']):
                p.data.copy_(swa_w)

    def zero_grad(self):
        """
        Sets the gradients of all optimized parameters to zero.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
