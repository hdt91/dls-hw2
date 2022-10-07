"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        s = self.scalar
        return out_grad * mul_scalar(power_scalar(x, s-1), s)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, y = node.inputs
        return divide(out_grad, y), multiply(out_grad, divide(mul_scalar(x, -1), power_scalar(y, 2)))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(a.ndim))
        if self.axes:
            x, y = self.axes
            axes[x], axes[y] = axes[y], axes[x]
        else:
            axes[-2:] = reversed(axes[-2:])

        return array_api.transpose(a, axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        # return broadcast_to(Tensor(1), x.shape)
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        # return broadcast_to(Tensor(1), x.shape)
        return reshape(out_grad, x.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        x_shape = [1] * len(out_grad.shape)
        for idx, d in enumerate(x.shape):
            x_shape[idx] = d

        # Find out which axis is broadcasted
        axes = tuple(idx for idx, (dx, dg) in enumerate(zip(x_shape, out_grad.shape)) if dx < dg)
        # print("foo", out_grad.dtype)
        return reshape(summation(out_grad, axes=axes), x.shape)
        # return broadcast_to(summation(out_grad, axes=axes), x.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        out_shape = list(x.shape)
        if self.axes:
            for axis in self.axes:
                out_shape[axis] = 1
        else:
            out_shape = [1] * len(x.shape)
        # return broadcast_to(Tensor(1), x.shape)
        return broadcast_to(reshape(out_grad, shape=out_shape), shape=x.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, y = node.inputs
        #print("X Y", x.shape, y.shape, out_grad.shape)
        dim_x, dim_y = len(x.shape), len(y.shape)
        dl_dx = matmul(out_grad, transpose(y))
        dl_dy = matmul(transpose(x), out_grad)
        dim_dlx, dim_dly = len(dl_dx.shape), len(dl_dy.shape)
        #print("DL_DX", dl_dx.shape, dl_dx.numpy()[0])
        #print("DL_DY", dl_dy.shape, dl_dy.numpy()[0])
        if dim_dlx != dim_x:
            dl_dx = summation(dl_dx, axes=tuple(range(dim_dlx - dim_x)))
        if dim_dly != dim_y:
            dl_dy = summation(dl_dy, axes=tuple(range(dim_dly - dim_y)))
        #dl_dy.numpy()[(0,) * (len(dl_dy.shape) - dim_y)])
        return dl_dx, dl_dy
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.multiply(a, -1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        #return broadcast_to(Tensor(-1), x.shape)
        return mul_scalar(out_grad, -1)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return out_grad * divide(broadcast_to(Tensor(1), x.shape), x)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return out_grad * exp(x)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x_data = node.inputs[0].realize_cached_data()
        grad = (x_data > 0) * array_api.float32(1.0)
        return out_grad * grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=self.axes, keepdims=True)
        exp = array_api.exp(Z - max_z, dtype=Z.dtype)
        S = array_api.sum(exp, axis=self.axes, keepdims=True, dtype=Z.dtype)
        result = array_api.log(S) + max_z
        self.exp = Tensor(exp)
        return result.squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        reshape_axes = []
        for axis in range(len(x.shape)):
            if self.axes is None or axis in self.axes:
                reshape_axes.append(1)
            else:
                reshape_axes.append(x.shape[axis])

        exp_x = self.exp
        exp_sum = summation(exp_x, axes=self.axes)
        exp_sum_out_grad = divide(out_grad, exp_sum)
        grad = exp_x * reshape(exp_sum_out_grad, shape=reshape_axes)
        return grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
