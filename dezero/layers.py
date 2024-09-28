import os
import numpy as np
import dezero.functions as F
from dezero.core import Parameter
from dezero.utils import pair
import weakref
from dezero import cuda


class Layer:
    def __init__(self):
        # save params belong to layer instance
        # _variable: name is meant for internal use only
        self._params = set()

    # setattr(__obj, __name, __value) -> setattr(x, "y", v) is equivalent to x.y = v
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + "/" + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def save_weights(self, save_path):
        self.to_cpu()  # only save as numpy array

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {
            key: param.data for key, param in params_dict.items() if param is not None
        }

        try:
            np.savez_compressed(save_path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(save_path):
                os.remove(save_path)  # 불완전한 파일의 생성 방지
            raise

    def load_weights(self, load_path):
        npz = np.load(load_path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# =============================================================================
# Linear / Conv2d / Deconv2d / MaxPool
# =============================================================================
class Linear(Layer):
    def __init__(self, out_size, bias=True, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size, self.out_size = in_size, out_size
        self.dtype = dtype

        # 초기값으로 None -> forward에서 가중치를 설정
        self.W = Parameter(None, name="W")
        if self.in_size is not None:  # in_size가 지정되지 않다면 나중으로 연기
            self._init_W()

        if bias:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")
        else:
            self.b = None

    def _init_W(self):
        I, O = self.in_size, self.out_size
        # 무작위 초기값 스케일: np.sqrt(1/I)
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # 데이터를 흘려보내는 시점에 가중치 초기화
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        y.name = "y"
        return y


class Conv2d(Layer):
    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        pad=0,
        nobias=False,
        dtype=np.float32,
        in_channels=None,
    ):
        super().__init__()
        self.in_channels = in_channels  # gray scale (1) or RGB, BGR (3) or CMYK(4)
        self.out_channels = out_channels  # number of filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name="W")

        # If in_channels is provided during the initialization,
        # _init_W() : the weights are immediately initialized
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)

        # The scale factor for initializing the weights, which is based on the input size
        # ensures the weights are initialized with a small value, which helps stabilize training
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class MaxPool(Layer):
    def __init__(self, kernel_size, stride=None, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        # Default stride : the same as kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.pad = pad

    def forward(self, x):
        y = F.pooling(x, self.kernel_size, self.stride, self.pad)
        y.name = "y"
        return y


# =============================================================================
# RNN / LSTM
# =============================================================================
class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, bias=False)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))

        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        H, I = hidden_size, in_size
        self.x2f = Linear(H, in_size=I)
        self.x2i = Linear(H, in_size=I)
        self.x2o = Linear(H, in_size=I)
        self.x2u = Linear(H, in_size=I)
        self.h2f = Linear(H, in_size=I, bias=False)
        self.h2i = Linear(H, in_size=I, bias=False)
        self.h2o = Linear(H, in_size=I, bias=False)
        self.h2u = Linear(H, in_size=I, bias=False)
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None  # memory cell

    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(x))
            i = F.sigmoid(self.x2i(x) + self.h2i(x))
            o = F.sigmoid(self.x2o(x) + self.h2o(x))
            u = F.tanh(self.x2u(x) + self.h2u(x))
        if self.c is None:
            c_new = i * u
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * F.tanh(c_new)
        self.h, self.c = h_new, c_new
        return h_new


# =============================================================================
# EmbedID / BatchNorm
# =============================================================================
class EmbedID(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name="W")

    def forward(self, x):
        y = self.W[x]
        return y


class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name="avg_mean")
        self.avg_var = Parameter(None, name="avg_var")
        self.gamma = Parameter(None, name="gamma")
        self.beta = Parameter(None, name="beta")

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def forward(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_norm(
            x, self.gamma, self.beta, self.avg_mean.data, self.avg_var.data
        )
