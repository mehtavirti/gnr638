from framework.cpp_backend import Tensor, conv2d_multi, relu3d, maxpool3d, flatten3d
import random


class Conv2DMulti:
    """Multi-channel 2D convolution"""
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # std = sqrt(2 / (in_channels * kernel_size * kernel_size))
        std = (2.0 / (in_channels * kernel_size * kernel_size)) ** 0.5
        
        print(f"         Init Conv({in_channels}->{out_channels})...", end='', flush=True)
        
        W_data = []
        for _ in range(out_channels):
            out_ch_data = []
            for _ in range(in_channels):
                kernel_data = []
                for _ in range(kernel_size):
                    kernel_data.append([random.gauss(0, std) for _ in range(kernel_size)])
                out_ch_data.append(kernel_data)
            W_data.append(out_ch_data)
        
        self.W = Tensor(W_data)
        self.b = Tensor([0.0 for _ in range(out_channels)])
        
        print(" Done")
    
    def __call__(self, x):
        return conv2d_multi(x, self.W, self.b)


class MaxPool2DMulti:
    """Multi-channel max pooling"""
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
    
    def __call__(self, x):
        return maxpool3d(x, self.pool_size)


class ReLUMulti:
    """Multi-channel ReLU"""
    def __call__(self, x):
        return relu3d(x)


class FlattenMulti:
    """Flatten multi-channel to 1D"""
    def __call__(self, x):
        return flatten3d(x)