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
    
    def set_creator(self, func) :
        self.creator = func
    
    def backward(self) :
        if self.grad is None :
            self.grad = np.ones_like(self.data)
            
        funcs = [self.creator]
        while funcs :
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            
            if x.creator is not None :
                funcs.append(x.creator)

class Function :
    def __call__(self, input) :
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self) # 출력 변수에 창조자 설정
        self.input = input # 입력 변수 보관
        self.output = output # 출력도 저장
        return output
    
    def forward(self, x) :
        raise NotImplementedError()
    
    def backward(self, gy) :
        raise NotImplementedError()

class Square(Function) :
    def forward(self, x) :
        return x ** 2
    
    def backward(self, gy) :
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function) :
    def forward(self, x) :
        return np.exp(x)
    
    def backward(self, gy) :
        x = self.input.data
        gx = np.exp(x) * gy
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

def as_array(x) :
    if np.isscalar(x) :
        return np.array(x)
    return x