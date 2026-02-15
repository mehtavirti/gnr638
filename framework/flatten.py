from framework.cpp_backend import flatten

class Flatten:
    def __call__(self, x):
        return flatten(x)