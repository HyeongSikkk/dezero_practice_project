import weakref
import contextlib
import numpy as np
import math

class Variable :
    __array_priority__ = 200
    def __init__(self, data, name = None) :
        # np.ndarray 만 취급
        if data is not None :
            if not isinstance(data, np.ndarray) :
                raise TypeError(f"{type(data)}은(는) 지원하지 않습니다.")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func) :
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad = False) :
        if self.grad is None :
            self.grad = np.ones_like(self.data)
            
        funcs = []
        seen_set = set()
        def add_func(f) :
            if f not in seen_set :
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x : x.generation)
                
        add_func(self.creator)
        while funcs :
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple) :
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs) :
                if x.grad is None :
                    x.grad = gx
                else :
                    x.grad = x.grad + gx
                
                if x.creator is not None :
                    add_func(x.creator)
                    
            if not retain_grad :
                for y in f.outputs :
                    y().grad = None
    
    def cleargrad(self) :
        self.grad = None
    
    @property
    def shape(self) :
        return self.data.shape
    
    @property
    def ndim(self) :
        return self.data.ndim
    
    @property
    def size(self) :
        return self.data.size
    
    @property
    def dtype(self) :
        return self.data.dtype
    
    def __len__(self) :
        return len(self.data)
    
    def __repr__(self) :
        if self.data is None :
            return "variable(None)"
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return "variable(" + p + ")"


class Function :
    def __call__(self, *inputs) :
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple): # 튜플이 아닐 경우
            ys = (ys,)
        outputs =[Variable(as_array(y)) for y in ys]
        
        if Config.enable_backdrop :            
            self.generation = max([x.generation for x in inputs])
            for output in outputs :
                output.set_creator(self) # 출력 변수에 창조자 설정
            self.inputs = inputs # 입력 변수 보관
            self.outputs = [weakref.ref(output) for output in outputs] # 출력도 저장
            
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs) :
        raise NotImplementedError()
    
    def backward(self, gys) :
        raise NotImplementedError()

class Config :
    enable_backdrop = True

class Square(Function) :
    def forward(self, x) :
        return x ** 2
    
    def backward(self, gy) :
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function) :
    def forward(self, x) :
        return np.exp(x)
    
    def backward(self, gy) :
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class Add(Function) :
    def forward(self, x0, x1) :
        y = x0 + x1
        return (y,)
    
    def backward(self, gy) :
        return gy, gy

class Mul(Function) :
    def forward(self, x0, x1) :
        y = x0 * x1
        return y
    
    def backward(self, gy) :
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

class Neg(Function) :
    def forward(self, x) :
        return -x
    
    def backward(self, gy) :
        return -gy
    
class Sub(Function) :
    def forward(self, x0, x1) :
        y = x0 - x1
        return y
    
    def backward(self, gy) :
        return gy, -gy

class Div(Function) :
    def forward(self, x0, x1) :
        y = x0 / x1
        return y
    
    def backward(self, gy) :
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1
    
class Pow(Function) :
    def __init__(self, c) :
        self.c = c
    
    def forward(self, x) :
        y = x ** self.c
        return y
    
    def backward(self, gy) :
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

class Sin(Function) :
    def forward(self, x) :
        y = np.sin(x)
        return y
    
    def backward(self, gy) :
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def numerical_diff(f, x, eps = 0.0001) :
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def square(x) :
    f = Square()
    return f(x)

def exp(x) :
    f = Exp()
    return f(x)

def add(x0, x1) :
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1) :
    # my code == x1 = as_array(x1)
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x) :
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1) :
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1) :
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1) :
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x, c) :
    return Pow(c)(x)

def sin(x) :
    return Sin()(x)

def my_sin(x, threshold = 0.0001) :
    y = 0
    for i in range(100000) :
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold :
            break
    return y

def as_array(x) :
    if np.isscalar(x) :
        return np.array(x)
    return x

def as_variable(obj) :
    if isinstance(obj, Variable) :
        return obj
    return Variable(obj)

@contextlib.contextmanager
def using_config(name, value) :
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try :
        yield
    finally :
        setattr(Config, name, old_value)

def no_grad() :
    return using_config('enable_backdrop', 'False')

Variable.__add__ = add
Variable.__radd__ = add #
Variable.__mul__ = mul
Variable.__rmul__ = mul #
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow