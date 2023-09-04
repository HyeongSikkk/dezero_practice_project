import weakref
import contextlib
import numpy as np

class Variable :
    def __init__(self, data) :
        # np.ndarray 만 취급
        if data is not None :
            if not isinstance(data, np.ndarray) :
                raise TypeError(f"{type(data)}은(는) 지원하지 않습니다.")
        self.data = data
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

class Function :
    def __call__(self, *inputs) :
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
    return Add()(x0, x1)

def as_array(x) :
    if np.isscalar(x) :
        return np.array(x)
    return x

@contextlib.contextmanager
def using_config(name, value) :
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try :
        yield
    finally :
        setattr(Config, name, old_value)