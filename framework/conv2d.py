from framework.cpp_backend import Tensor, conv2d

class Conv2D:
    def __init__(self, kernel_size):
        import random
        # Initialize kernel weights as C++ Tensor
        self.W = Tensor([
            [random.uniform(-0.1, 0.1) for _ in range(kernel_size)]
            for _ in range(kernel_size)
        ])
        self.b = Tensor([0.0])

    def __call__(self, x):
      
        result = conv2d(x, self.W, self.b)
            
        return result