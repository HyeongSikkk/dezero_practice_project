from dezero.core import (
    Function, 
    as_variable,
)
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


class Tanh(Function) :
    def forward(self, x) :
        y = np.tanh(x)
        return y
    
    def backward(self, gy) :
        y = self.outputs[0]()
        gx = gy * (1 - y*y)
        return gx
    
def tanh(x) :
    return Tanh()(x)


class Reshape(Function) :
    def __init__(self, shape) :
        self.shape = shape
        
    def forward(self, x) :
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy) :
        return reshape(gy, self.x_shape)

def reshape(x, shape) :
    if x.shape == shape :
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)