import numpy as np


class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []  # preprocessing: weight decay, gradient clipping

    def setup(self, target):  # target: Model or Layer
        self.target = target
        return self

    def update(self):  # update all parameters
        # None 이외의 매개변수를 리스트에 모아둔다
        params = [p for p in self.target.params() if p.grad is not None]

        # 전처리(optional)
        for f in self.hooks:
            f(params)

        # update a single parameter
        for param in params:
            self.update_one(param)

    def update_one(self, param):  # 파라미터 하나씩 처리
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


# =============================================================================
# SGD / Momentum / AdaGrad / RMSProp / AdaDelta / Adam
# =============================================================================
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class Momentum(Optimizer):
    # v <- a*v - lr*g and W <- W(param) + v where a: momentum (0<a<1)
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}  # empty dict

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)  # shape(W) == shape(v)

        v = self.vs[v_key]

        # v = self.momentum*v - self.lr*param.grad.data
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):  # Adaptive Gradient
    # h <- h + g*g
    # w <- w - lr*g/sqrt(h+eps)
    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = np.zeros_like(param.data)

        h = self.hs[h_key]
        h += param.grad.data * param.grad.data
        param.data -= self.lr * param.grad.data / (np.sqrt(h) + self.eps)


class RMSProp(Optimizer):
    def __init__(self, lr=0.1, decay_rate=0.95, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.rho = decay_rate
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = np.zeros_like(param.data)

        h = self.hs[h_key]
        h *= self.rho
        h += (1 - self.rho) * param.grad.data * param.grad.data

        param.data -= self.lr * param.grad.data / (np.sqrt(h) + self.eps)


class AdaDelta(Optimizer):
    # newton method approximation -> lr ~
    def __init__(self, decay_rate=0.95, eps=1e-6):
        super().__init__()
        self.rho = decay_rate
        self.eps = eps
        self.hs = {}
        self.steps = {}

    def update_one(self, param):
        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = np.zeros_like(param.data)
            self.steps[h_key] = np.zeros_like(param.data)

        h = self.hs[h_key]
        step = self.steps[h_key]
        g = param.grad.data

        h = self.rho * h + (1 - self.rho) * g * g
        param.data -= np.sqrt((step + self.eps) / (h + self.eps)) * g
        step = self.rho * step + (1 - self.rho) * g * g


class Adam(Optimizer):
    # RMSProp + Momentum
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.alpha = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iter = 0
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.iter += 1
        super().update()

    @property
    def lr(self):
        beta1 = 1 - np.pow(self.beta1, self.iter)
        beta2 = 1 - np.pow(self.beta2, self.iter)
        return self.alpha * np.sqrt(beta2) / beta1

    def update_one(self, param):
        key = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)

        m = self.ms[key]
        v = self.vs[key]
        g = param.grad.data

        m += (1 - self.beta1) * (g - m)
        v += (1 - self.beta2) * (g * g - v)
        param.data -= self.lr * m / (np.sqrt(v) + self.eps)


# =============================================================================
# Hook functions
# =============================================================================
class WeightDecay:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data


class ClipGrad:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data**2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad.data *= rate


class FreezeParam:
    def __init__(self, *layers):
        self.freeze_params = []
        for l in layers:
            if isinstance(l, Parameter):
                self.freeze_params.append(l)
            else:
                for p in l.params():
                    self.freeze_params.append(p)

    def __call__(self, params):
        for p in self.freeze_params:
            p.grad = None
