from dezero.core import Function
import numpy as np

class Sin(Function) :
    def forward(self, x) :
        y = np.sin(x)
        return y
    
    def backward(self, gy) :
        x, = self.inputs # x = self.inputs[0] 과 같음
        gx = gy * cos(x)
        return gx

def sin(x) :
    return Sin()(x)


class Cos(Function) :
    def forward(self, x) :
        y = np.cos(x)
        return y
    
    def backward(self, gy) :
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x) :
    return Cos()(x)