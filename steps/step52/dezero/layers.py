from dezero.core import Parameter
from dezero import functions as F
import weakref
import numpy as np
from dezero import cuda

class Layer :
    def __init__(self) :
        self._params = set()
    
    # 인스턴스 변수를 설정하되, 만약 변수가 Parameter 일 경우 _params에 추가함    
    def __setattr__(self, name, value) :
        if isinstance(value, (Parameter, Layer)) :
            self._params.add(name)
        super().__setattr__(name, value)
    
    def __call__(self, *inputs) :
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple) :
            outputs = (outputs,)
        
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def to_cpu(self) :
        for param in self.params() :
            param.to_cpu()
    
    def to_gpu(self) :
        for param in self.params() :
            param.to_gpu()
    
    def forward(self, inputs) :
        raise NotImplementedError()
    
    def params(self) :
        for name in self._params :
            obj =  self.__dict__[name]
            
            if isinstance(obj, Layer) : # Layer에서 매개변수 추출
                yield from obj.params() # Layer 인스턴스인 경우 해당 인스턴스로 제너레이터 생성
            else :
                yield obj
            
    def clear_grads(self) :
        for param in self.params() :
            param.clear_grad()
            

class Linear(Layer) :
    def __init__(self, out_size, nobias = False, dtype = np.float32, in_size = None) :
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name = 'W')
        if self.in_size is not None :
            self._init_W()
            
        if nobias :
            self.b = None
        else :
            self.b = Parameter(np.zeros(out_size, dtype = dtype), name = 'b')
    
    def _init_W(self, xp = np) :
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
        
    def forward(self, x) :
        # 데이터를 흘려보내는 시점에 가중치 초기화
        if self.W.data is None :
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)
            
        y = F.linear(x, self.W, self.b)
        return y