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
        # Add these debug lines:
        # print(f"DEBUG Conv2D - input has data_mat: {hasattr(x, 'data_mat')}, len: {len(x.data_mat) if hasattr(x, 'data_mat') else 'N/A'}")
        # print(f"DEBUG Conv2D - W shape: {len(self.W.data_mat)}x{len(self.W.data_mat[0])}")
        # print(f"DEBUG Conv2D - b: {self.b.data_vec}")
        
        # Use C++ backend conv2d operation
        result = conv2d(x, self.W, self.b)
        
        # print(f"DEBUG Conv2D - output has data_mat: {hasattr(result, 'data_mat')}, len: {len(result.data_mat) if hasattr(result, 'data_mat') else 'N/A'}")
        
        return result