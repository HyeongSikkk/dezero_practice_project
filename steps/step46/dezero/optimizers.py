class Optimizer :
    def __init__(self) :
        self.target = None
        self.hooks = []
        
    def setup(self, target) :
        self.target = target
        return self
    
    def update(self) :
        # None 이외의 매개변수를 리스트에 모아둠
        params = [p for p in self.target.params() if p.grad is not None]
        for f in self.hooks :
            f(params)
        
        for param in params :
            self.update_one(param)
    
    def update_one(self, param) :
        raise NotImplementedError()
    
    def add_hook(self, f) :
        self.hooks.append(f)


class SGD(Optimizer) :
    def __init__(self, lr = 0.01) :
        super().__init__()
        self.lr = lr
    
    def update_one(self, param) :
        param.data -= self.lr * param.grad.data 