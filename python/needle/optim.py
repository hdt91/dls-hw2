"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

        self.counter = 0

    def step(self):
        self.counter += 1
        ### BEGIN YOUR SOLUTION
        for idx, param in enumerate(self.params):
            grad = param.grad.detach()
            if self.weight_decay > 0.0:
                grad += param.detach() * self.weight_decay
            scaled_grad = ndl.ops.mul_scalar(grad, 1 - self.momentum)
            if idx not in self.u:
                self.u[idx] = scaled_grad
            else:
                self.u[idx] = ndl.add(ndl.ops.mul_scalar(self.u[idx].detach(), self.momentum), scaled_grad).detach()

            param = ndl.add(param.detach(), ndl.mul_scalar(self.u[idx], -self.lr))
            self.params[idx].data = ndl.Tensor(param)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        decay_beta1 = (1 - self.beta1 ** self.t)
        decay_beta2 = (1 - self.beta2 ** self.t)

        for idx, param in enumerate(self.params):
            grad = param.grad.detach()
            if self.weight_decay > 0.0:
                grad += (param.detach() * self.weight_decay)

            scaled_grad_1 = ndl.ops.mul_scalar(grad, 1 - self.beta1)
            scaled_grad_2 = ndl.ops.mul_scalar(grad ** 2, 1 - self.beta2)
            if idx not in self.m:
                self.m[idx] = scaled_grad_1
                self.v[idx] = scaled_grad_2
            else:
                self.m[idx] = ndl.add(ndl.ops.mul_scalar(self.m[idx].detach(), self.beta1), scaled_grad_1).detach()
                self.v[idx] = ndl.add(ndl.ops.mul_scalar(self.v[idx].detach(), self.beta2), scaled_grad_2).detach()

            m = self.m[idx] / decay_beta1
            v = self.v[idx] / decay_beta2

            update = (m * -self.lr) / (v ** 0.5 + self.eps)
            param = param.detach() + update

            self.params[idx].data = ndl.Tensor(param)
        ### END YOUR SOLUTION
