from framework.cpp_backend import maxpool2d

class MaxPool2D:
    def __init__(self, pool_size=2):
        self.p = pool_size

    def __call__(self, x):
        return maxpool2d(x, self.p)