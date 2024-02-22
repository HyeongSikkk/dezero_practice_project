class Variable :
    def __init__(self, data) :
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func) :
        self.creator = func
    
    def backward(self) :
        f = self.creator
        if f is not None :
            x = f.input # 함수의 입력값 호출
            x.grad = f.backward(self.grad)
            x.backward()
            
        
class Function :
    def __call__(self, input) :
        x = input.data
        y = self.forward(x) # 구체적인 계산은 forward에서 이루어짐
        output = Variable(y)
        # 출력값에 출처 표기
        output.set_creator(self)
        
        # 함수 입출력 변수 표기
        self.input = input
        self.output = output
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

def numerical_diff(f, x, eps = 1e-4) :
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)