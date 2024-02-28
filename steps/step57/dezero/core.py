import numpy as np
import weakref
import dezero

class Config :
    enable_backprop = True
    train = True


try :
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError :
    array_types = (np.ndarray)
    
class Variable :
    __array_priority__ = 200 # Variable 인스턴스의 연산 우선순위 높이기
    
    def __init__(self, data, name = None) :
        if data is not None :
            if not isinstance(data, array_types) :
                raise TypeError(f"{type(data)}은(는) 지원하지 않습니다.")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    
    
    def to_cpu(self) :
        if self.data is not None :
            self.data = dezero.cuda.as_numpy(self.data)
            
    def to_gpu(self) :
        if self.data is not None :
            self.data = dezero.cuda.as_cupy(self.data)
    
    def set_creator(self, func) :
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad = False, create_graph = False) :
        if self.grad == None :
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(np.ones_like(self.data))
        
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
            
            with using_config('enable_backprop', create_graph) :
                gxs = f.backward(*gys) # 메인 backward
            
                if not isinstance(gxs, tuple) :
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs) :
                    if x.grad is None :
                        x.grad = gx
                    else :
                        x.grad = x.grad + gx

                    if x.creator != None :
                        add_func(x.creator)

                if not retain_grad :
                    for y in f.outputs :
                        y().grad = None
                    
    
    def clear_grad(self) :
        self.grad = None
        
    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)
    
    def reshape(self, shape) :
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
    
    def sum(self, axis = None, keepdims = False) :
        return dezero.functions.sum(self, axis, keepdims)
    
    @property
    def T(self) :
        return dezero.functions.transpose(self)
    
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
           return 'variable(None)'
       p = str(self.data).replace('\n', '\n' + ' ' * 9)
       return 'variable(' + p +')'


class Parameter(Variable) :
    pass

class Function :
    def __call__(self, *inputs) :
        inputs = [as_variable(x) for x in inputs]
        
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 구체적인 계산은 forward에서 이루어짐
        if not isinstance(ys, tuple) :
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop :
        
            self.generation = max([x.generation for x in inputs]) # 입력값들 중 제일 큰 세대값
            # 출력값에 출처 표기
            for output in outputs :
                output.set_creator(self)

            # 함수 입출력 변수 표기
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        
        # 리스트의 원소가 1개인 경우 원소를 반환한다.
        return outputs if len(outputs) > 1 else outputs[0] 
    
    def forward(self, xs) :
        raise NotImplementedError()
    
    def backward(self, gys) :
        raise NotImplementedError()
 

class Add(Function) :
    def forward(self,x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    def backward(self, gy) :
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape :
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

def add(x0, x1) :
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Add()(x0,x1)


class Neg(Function) :
    def forward(self, x) :
        return -x
    
    def backward(self, gy) :
        return -gy

def neg(x) :
    return Neg()(x)


class Sub(Function) :
    def forward(self, x0, x1) :
        y = x0 - x1
        return y
    
    def backward(self, gy) :
        return gy, -gy

def sub(x0, x1) :
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)
    
def rsub(x0, x1) :
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


class Mul(Function) :
    def forward(self, x0, x1) :
        y = x0*x1
        return y
    
    def backward(self, gy) :
        x0, x1 = self.inputs
        return gy*x1, gy*x0

def mul(x0, x1) :
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Div(Function) :
    def forward(self, x0, x1) :
        y = x0/x1
        return y
    
    def backward(self, gy) :
        x0, x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy * (-x0/x1**2)
        return gx0, gx1

def div(x0, x1) :
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)

def rdiv(x0, x1) :
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function) :
    def __init__(self, c) :
        self.c = c
    
    def forward(self, x) :
        c = self.c
        y = x ** c
        return y
    
    def backward(self, gy) :
        x = self.inputs[0]
        c = self.c
        gx = c * x**(c-1) * gy
        return gx
    
    
def pow(x, c) :
    return Pow(c)(x)


def as_array(x, array_module = np) :
    if np.isscalar(x) :
        return array_module.array(x)
    return x


def as_variable(obj) :
    if isinstance(obj, Variable) :
        return obj
    return Variable(obj)


import contextlib

@contextlib.contextmanager
def using_config(name, value) :
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try :
        yield
    finally :
        setattr(Config, name, old_value)


def no_grad() :
    return using_config('enable_backprop', False)

def test_mode() :
    return using_config('train', False)

def setup_variable() :
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dezero.functions.get_item