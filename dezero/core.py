import dezero
import numpy as np
import heapq
import weakref
from contextlib import contextmanager


# =============================================================================
# Config
# =============================================================================
class Config:  # 순전파만 수행할지, 역전파도 수행할지 설정
    # 클래스 속성: 클래스에 속하며 모든 인스턴스와 공유 : 설정데이터는 단 한군데 존재하도록
    enable_backprop = True
    train = True


@contextmanager
def using_config(name, value=False):
    #  np.array(1) == getattr(np, "array")(1) : 차이점 -> getattr의 첫번째 인자(property name)는 str을 받는다
    old_value = getattr(Config, name)  # True
    # test set 평가시 setattr를 이용하여 Config.enable_backprop = False로 설정
    setattr(Config, name, value)

    try:
        yield
    finally:  # 예외발생 상관없이 무조건 실행
        # with 절 종료시에 원래 값 Config.enable_backprop = True로 되돌리기
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


def test_mode():
    return using_config("train", False)


# =============================================================================
# Variable / Function
# =============================================================================

try:
    import cupy

    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = np.ndarray


class Variable:
    # ndarray 연산자 우선순위 변경
    __array_priority__ = 200

    def __init__(self, data, name=None):
        # None or ndarray is acceptable for input data
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError("{}은(는) 지원하지 않습니다.".format(type(data)))

        self.data = data
        self.name = name
        self.grad = None  # backpropagation시에 미분값을 저장할수 있도록
        self.creator = None  # x -> f -> y : f is the creator of y
        self.generation = 0  # update generation when forward

    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    def reshape(self, *shape):
        # ex: x.reshape((a,b)) or x.reshape([a,b])
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        # to prevent circular import: not F.reshape but dezero.functions.reshape
        return dezero.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None

        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def matmul(self, other):
        return dezero.functions.matmul(self, other)

    def dot(self, other):
        return dezero.functions.matmul(self, other)

    def max(self, axis=None, keepdims=False):
        return dezero.functions.max(self, axis, keepdims)

    def min(self, axis=None, keepdims=False):
        return dezero.functions.min(self, axis, keepdims)

    # 연산자 오버로드
    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __mul__(self, other):  # same as Variable.__mul__ = mul, x*3.0
        return mul(self, other)

    def __rmul__(self, other):  # ex: 3.0 * x
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __pow__(self, c):
        return pow(self, c)

    def __getitem__(self, idx):
        return dezero.functions.get_item(self, idx)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 부모함수 + 1

    def backward(self, retain_grad=False, create_graph=False):
        # retain_grad : option whether intermediate gradient would be saved or deleted.
        # backpropagation 자동화
        # 1. 미분값의 creator(해당 함수)를 불러온다.
        # 2. 해당 함수의 입력값을 가져온다.
        # 3. 해당 함수의 backward 호출한다

        # self.data 와 동일한 shape 및 data type으로 gradient 값 1로 초기화 (dy/dy = 1)
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs, (-f.generation, f))
                seen_set.add(f)

        add_func(self.creator)

        while funcs:
            # 함수를 불러온다
            f = heapq.heappop(funcs)[1]
            # 함수의 입,출력값을 불러온다
            # 1. 출력변수인 outputs 에 담겨있는 미분값들을 리스트에 담는다.
            # 수정전 순환참조: gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]  # weakref - f.outputs

            with using_config("enable_backprop", create_graph):
                # 2. 함수 f의 backward 메서드를 호출
                gxs = f.backward(*gys)  # unpack list of gys
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                # backpropagation으로 전파되는 미분값을 Variable 인슨턴스 변수 grad에 저장 (f.inputs[i] = gxs[i]로 업데이트)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx  # inplace memory문제: x.grad += gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:  # remove intermediate gradient
                for y in f.outputs:
                    y().grad = None  # y : weakref => y(): data of y

    def cleargrad(self):  # grad 초기화
        self.grad = None

    def unchain(self):
        self.creator = None

    def unchain_backward(self):
        # 호출된 변수에서 시작하여 계산변수를 거슬러 올라가며 마주치는 모든 변수의 unchain 메소드를 호출
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()


class Parameter(Variable):
    def __repr__(self):
        if self.data is None:
            return "parameter(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "parameter(" + p + ")"


class Function:  # 모든 함수에 구현되는 공통기능만을 구현 : base class
    def __call__(self, *inputs):  # 클래스의 인스턴스를 함수처럼 호출
        # input: Variable instance
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        # unpack xs list and foward
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        # make sure that y should be 'ndarray'
        outputs = [Variable(as_array(y)) for y in ys]

        # 역전파가 필요한 경우(learning), 순전파만 - 추론(evaluation)
        # config.enable_backprop == True: Variable과 Function 연결 - graph
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            # output variable에 creator 설정
            for output in outputs:
                output.set_creator(self)

            # forward 호출시 건네받은 variable 인스턴스 유지
            self.inputs = inputs  # 입력변수를 기억한다 - when backpropagation happens, forward results are required.
            # memory management: self.outputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        # 상위 클래스를 설계할 때, 하위 클래스에서 반드시 오버라이드하여 상세하게 구현해야 하는 메소드를 명시하고자
        raise NotImplementedError("forward 메서드를 구현해야합니다.")

    def backward(self, gys):
        raise NotImplementedError("backward 메서드를 구현해야합니다.")

    def __lt__(self, other):
        return self.generation < other.generation


def as_variable(x):
    if isinstance(x, Variable):
        return x
    return Variable(x)


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


# =============================================================================
# 기본 연산자 (+, -, *, /, **)
# =============================================================================
class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 - x1

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return (gx0, gx1)


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1

    def backward(self, gy):
        gx0, gx1 = gy, gy
        # broadcasting case : broadcast_to(forward/backward) <-> sum_to(backward/forward)
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return (gx0, gx1)


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = x1 * gy, x0 * gy
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy / x1, gy * (-x0 / x1**2)
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, a):
        self.a = a

    def forward(self, x):
        return x**self.a

    def backward(self, gy):
        (x,) = self.inputs
        gx = self.a * x ** (self.a - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def mul(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


def add(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


def div(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x1, x0)
