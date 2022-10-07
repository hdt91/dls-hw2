"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        if bias:
            self.bias = Parameter(ops.transpose(init.kaiming_uniform(out_features, 1)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # X shape is (N, H_in). H_in = in_features
        # Output should be (N, H_out). H_out = out_features
        N = X.shape[0]
        if len(X.shape) == 2:
            bc_shape = (X.shape[0], self.out_features)
            bias_shape = (1, self.out_features)
        else:
            bc_shape = (X.shape[0], X.shape[1], self.out_features)
            bias_shape = (1, 1, self.out_features)
        return X @ self.weight + ops.broadcast_to(ops.reshape(self.bias, bias_shape), shape=bc_shape)
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        flat = int(np.prod(X.shape) / X.shape[0])
        return ops.reshape(X, shape=(X.shape[0], flat))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        for module in self.modules:
            y = module.forward(y)
        return y
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        m, n = logits.shape
        y_one_hot = init.one_hot(n, y)
        losses = ops.logsumexp(logits, axes=(1,)) - ops.summation(ops.multiply(logits, y_one_hot), axes=(1,))
        return ops.summation(losses) / (m * 1.0) # Average loss on the batch
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m, dim = x.shape[0], self.dim
        # print("m - dim", m, dim)
        mean_x = ops.divide_scalar(ops.summation(x, axes=(0,)), m)
        mean_x_reshape = ops.broadcast_to(ops.reshape(mean_x, shape=(1, dim)), shape=(m, dim))
        var_x = ops.divide_scalar(ops.summation(ops.power_scalar(x - mean_x_reshape, 2), axes=(0,)), m)
        var_x_reshape = ops.broadcast_to(ops.reshape(var_x, shape=(1, dim)), shape=(m, dim))
        normalized_x = ops.divide(x - mean_x_reshape, ops.power_scalar(ops.add_scalar(var_x_reshape, self.eps), 0.5))

        self.running_mean = ops.mul_scalar(self.running_mean, 1 - self.momentum) + ops.mul_scalar(mean_x, self.momentum)
        self.running_var = ops.mul_scalar(self.running_var, 1 - self.momentum) + ops.mul_scalar(var_x, self.momentum)

        weight = ops.broadcast_to(ops.reshape(self.weight, shape=(1, dim)), shape=(m, dim))
        bias = ops.broadcast_to(ops.reshape(self.bias, shape=(1, dim)), shape=(m, dim))
        # print("normalized_x", normalized_x)
        # print("WWW", self.weight)
        # print(weight)
        # print("BBB", self.bias)
        # print(bias)
        # print("RETURN", ops.add(ops.multiply(normalized_x, weight), bias))
        return ops.add(ops.multiply(normalized_x, weight), bias)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m = x.shape[0]
        mean_x = ops.divide_scalar(ops.summation(x, axes=(1,)), self.dim)
        mean_x_reshape = ops.broadcast_to(ops.reshape(mean_x, shape=(m, 1)), shape=(m, self.dim))
        var_x = ops.divide_scalar(ops.summation(ops.power_scalar(x - mean_x_reshape, 2), axes=(1,)), self.dim)
        var_x_reshape = ops.broadcast_to(ops.reshape(var_x, shape=(m,1)), shape=(m, self.dim))
        normalized_x = ops.divide(x - mean_x_reshape, ops.power_scalar(ops.add_scalar(var_x_reshape, self.eps), 0.5))

        return ops.multiply(normalized_x, self.weight) + self.bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x

        mask = init.randb(*x.shape, p=(1-self.p))
        return ops.divide_scalar(x * mask, 1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.add(self.fn(x), x)
        ### END YOUR SOLUTION



