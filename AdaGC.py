# python3.11
# -*- coding: utf-8 -*-
# @Time    : 2023/12/28 15:04
# @Author  : Feng Su
# @File    : AdaGC.py
# @Software: PyCharm

import math
import torch
from torch.optim.optimizer import Optimizer
from types_2 import Betas2, OptFloat, OptLossClosure, Params
import time

__all__ = ('AdaGC',)

class AdaGC(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        beta3:float= 0.999,
        eps: float = 1e-8,

    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if not 0.0 <= beta3 < 1.0:
            raise ValueError('Invalid beta3 parameter: {}'.format(beta3))

        defaults = dict(lr=lr,betas=betas,eps=eps,beta3=beta3)
        super(AdaGC, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaGC does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )

                state = self.state[p]
                beta1, beta2 = group['betas']

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_s'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_lr'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['PG'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_grad_norm'] = torch.zeros_like(torch.norm(grad,p=2))

                PG, exp_avg, exp_avg_var=state['PG'], state['exp_avg'],state['exp_avg_var']
                exp_avg_s, exp_avg_lr= state['exp_avg_s'], state['exp_avg_lr']
                step = state['step']
                lr = group['lr']
                beta3 = group['beta3']
                step += 1

                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                bia=grad-PG
                exp_avg_s.mul_(beta1).add_(bia,alpha=1-beta1)
                nt = exp_avg.mul(beta2).add(exp_avg_s,alpha = 1-beta2)

                exp_avg_var.mul_(beta2).add_(bia.mul(bia),alpha=1-beta2)
                denom = exp_avg_var.sqrt().add(group['eps'])

                step_size = lr
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom)

                exp_avg_lr.mul_(beta3).add_(step_size, alpha=1 - beta3)
                step_size = torch.min(step_size, exp_avg_lr)

                step_size.mul_(nt)

                state['PG'] =nt
                p.data.add_(-step_size)

        return loss
