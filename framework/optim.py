class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr

    def step(self):
        for p in self.params:
            p.sgd_step(self.lr)   

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()        