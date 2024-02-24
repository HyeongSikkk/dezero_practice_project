import numpy as np
import weakref


class Config :
    enable_backdrop = True


class Variable :
    def __init__(self, data, name = None) :
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func) :
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad = False) :
        if self.grad == None :
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
                
                if x.creator != None :
                    add_func(x.creator)
            
            if not retain_grad :
                for y in f.outputs :
                    y().grad = None
                    
    
    def clear_grad(self) :
        self.grad = None
    
    def __add__(self, other) :
        return Add()(self, other)
    
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
            
        
class Function :
    def __call__(self, *inputs) :
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 구체적인 계산은 forward에서 이루어짐
        if not isinstance(ys, tuple) :
            ys = (ys,)
        outputs = [Variable(as_array(y.data)) for y in ys]
        
        if Config.enable_backdrop :
        
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
        

class Square(Function) :
    def forward(self, x) :
        return x ** 2
    
    def backward(self, gy) :
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
        

def square(x) :
    f = Square()
    return f(x)

class Exp(Function) :
    def forward(self, x) :
        return np.exp(x)
    
    def backward(self, gy) :
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

def exp(x) :
    f = Exp()
    return f(x)

class Add(Function) :
    def forward(self,x0, x1):
        x0, x1
        y = x0 + x1
        return y
    
    def backward(self, gy) :
        return gy, gy

def add(*xs) :
    f = Add()
    x0, x1 = xs
    return f(x0, x1)

def numerical_diff(f, x, eps = 1e-4) :
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

def as_array(x) :
    if np.isscalar(x) :
        return np.array(x)
    return x


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
    return using_config('enable_backdrop', False)