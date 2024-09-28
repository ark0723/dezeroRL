import numpy as np
from dezero.core import Function, as_variable, Variable, as_array
from dezero import utils, cuda
import math
import dezero


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.sin(x)

    def backward(self, gy):
        (x,) = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


# taylor series: non-polynomial -> polynomial approximation
def sin_approx(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        gx = gy * (-sin(x))
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y**2)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        return gy * y


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.log(x)

    def backward(self, gy):
        (x,) = self.inputs
        return gy / x


def log(x):
    return Log()(x)


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        # x: ndarray
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        # gy: Variable instance
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):  # x: ndarray
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):  # gy: Variable instance
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        (x,) = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(x)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)
        if xp is np:
            # np.ufunc.at(a, indices, b=None) : perform ufunc on 'a' after indexing/slicing
            np.add.at(gx, self.slices, gy)  # gx += gy after slicing using self.slicing
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, x):
        return get_item(x, self.slices)


def get_item(x, slices):
    return GetItem(slices)(x)


def expand_dims(x, axis):
    x = as_variable(x)
    # x.shape : tuple
    shape = list(x.shape)
    # list.insert(a,b): index a에 b 를 삽입
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    # x.shape =  (n, a, b) : n is the # of data or batch size, (a,b): data shape
    return reshape(x, (x.shape[0], -1))


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / mse / linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape  # target shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)


mean = average


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gW = matmul(x.T, gy)
        gx = matmul(gy, W.T)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None  # remove the data of t
    return y


# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================


class Sigmoid(Function):
    def forward(self, x):
        """
        why I used np.tanh instead of np.exp?
        1/(1+np.exp(-x)) = (tanh(x/2) + 1)/2

        For large negative values of x, np.exp(-x) can cause an overflow, making the computation prone to returning 0 due to floating-point precision limits.
        For large positive values of x, np.exp(-x) approaches 0, leading to potential underflow and loss of precision.
        """
        xp = cuda.get_array_module(x)
        return xp.tanh(0.5 * x) * 0.5 + 0.5

    def backward(self, gy):
        # f'(x) = f(x)*(1-f(x))
        y = self.outputs[0]()
        return gy * y * (1 - y)


def sigmoid(x):
    return Sigmoid()(x)


class Relu(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.maximum(0, x)

    def backward(self, gy):
        (x,) = self.inputs
        # mask = 1 if x.data > 0 else 0
        mask = x.data > 0
        gx = mask * gy
        return gx


def relu(x):
    return Relu()(x)


class LeakyRelu(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy):
        (x,) = self.inputs
        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx = gy * mask
        return gx


def leaky_relu(x, slope=0.01):
    return LeakyRelu(slope)(x)


class Softmax(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        # to prevent overflow
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        # gx = y*(gy - sumdx) where sumdx = sum_j(y_j*gy_j)
        y = self.outputs[0]()  # weakref
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


class LogSoftmax(Function):
    # log(softmax(x))
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        # utils.logsumexp(x, self.axis) = log(e^x_1 + e^x_2 + ... + e^x_i)
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)


# =============================================================================
# loss function: mse(mean squared error) / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================
class MSE(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff**2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mse(x0, x1):
    return MSE()(x0, x1)


def softmax_cross_entropy_simple(x, t):
    """
    loss = - sum_k(t_k*log(P_k))
    where P_k is the output of softmax(x)
    t_k : real y's label

    example)
    x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
    t = np.array([2,0,1,0])
    """
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]  # number of data
    p = softmax(x)
    # to prevent log(0), 1e-15 <= p <=1 : clip(x, x_min, x_max)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        # log(p) = log(softmax(x)) = x - log(sum(e^x_i))
        # loss = -sum(t_k * log(p_k)) / N
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        loss = -log_p.sum() / N
        return loss

    def backward(self, gy):
        # gx = -(onehot_t - y) / N where y = softmax(x)
        x, t = self.inputs
        N, num_class = x.shape

        gx = softmax(x)
        # t: convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(num_class, dtype=t.dtype)[t.data]
        gx = (gx - t_onehot) * gy / N
        return gx


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def sigmoid_cross_entropy(x, t):
    # loss = -sum_i(t_i*log(f(x_i))) where f(x) = 1/(1+e^(-x)
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(t)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


# =============================================================================
# accuracy / dropout / batch_norm / embed_id
# =============================================================================
def accuracy(y, y_true):
    y, y_true = as_variable(y), as_variable(y_true)
    pred = y.data.argmax(axis=1).reshape(y_true.shape)
    acc = (pred == y_true.data).mean()
    return Variable(as_array(acc))


class Dropout(Function):
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, x):
        # x : np.ndarray or cp.ndarray
        if dezero.Config.train:
            xp = cuda.get_array_module(x)
            mask = xp.random.rand(*x.shape) > self.dropout_ratio
            scale = xp.array(1 - self.dropout_ratio).astype(x.dtype)
            y = x * mask / scale
            return y
        else:
            return x

    def backward(self, gy):
        return gy


def dropout(x, droupout_ratio=0.5):
    return Dropout(droupout_ratio)(x)


class BatchNorm(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        # y_i <- gamma*x_i + beta where x over a mini-batch
        # x_i <- x_i - mean(Batch) / sqrt(var(Batch) + eps)
        # gamma and beta: parameters to learn
        assert x.ndim == 2 or x.ndim == 4

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            # (N, C, H, W) -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = cuda.get_array_module(x)

        if dezero.Config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            # x.size = N*H*W*C, gamma.size
            # m : mini-batch size
            m = x.size // gamma.size
            # compute unbiased variance estimation
            s = m - 1.0 if m - 1.0 > 1.0 else 1.0
            # unbiased estimation Var(x) <- (m / (m -1))*Var(Batch)
            adjust = m / s
            # The decay factor controls how much of the old average to keep versus the new mean.
            self.avg_mean *= self.decay
            #  Incorporate the new batch mean into the running average.
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            # Store the inverse standard deviation for use in backpropagation
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std
        y = gamma * xc + beta

        if x_ndim == 4:
            # (N*H*W, C) -> (N, C, H, W)
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return gamma * self.inv_std


def batch_norm(x, gamma, beta, mean, var, decay=0.9, eps=2e-5):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


def embed_id(x, W):
    return W[x]


# =============================================================================
# max / min / clip
# =============================================================================
class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return x.max(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = x.data == y.data
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    def forward(self, x):
        return x.min(axis=self.axis, keepdims=self.keepdims)


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    # x = x_min if x <= x_min, x = x_max if x >= x_max, else x
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        mask = self.x_min < x.data < self.x_max
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


# =============================================================================
# conv2d / col2im / im2col / basic_math
# =============================================================================
from dezero.functions_conv import conv2d
from dezero.functions_conv import deconv2d
from dezero.functions_conv import conv2D_simple
from dezero.functions_conv import im2col
from dezero.functions_conv import col2im
from dezero.functions_conv import pooling_simple
from dezero.functions_conv import Pooling2DGrad
from dezero.functions_conv import pooling
from dezero.functions_conv import average_pooling
from dezero.core import add
from dezero.core import sub
from dezero.core import rsub
from dezero.core import mul
from dezero.core import div
from dezero.core import neg
from dezero.core import pow
