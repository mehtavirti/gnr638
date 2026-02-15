from framework.cpp_backend import Tensor, linear

class Linear:
    def __init__(self, in_features, out_features):
        # Initialize weights as C++ Tensors
        import random
        self.W = Tensor([
            [random.uniform(-0.1, 0.1) for _ in range(out_features)] 
            for _ in range(in_features)
        ])
        self.b = Tensor([0.0 for _ in range(out_features)])

    def __call__(self, x):
        # Use C++ backend linear operation
        return linear(x, self.W, self.b)