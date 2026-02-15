import sys
import os

cpp_path = os.path.join(os.path.dirname(__file__), "cpp")
sys.path.insert(0, cpp_path)

import backend as _backend

# Original functions
Tensor = _backend.Tensor
add = _backend.add
mul = _backend.mul
sum = _backend.sum
matmul = _backend.matmul
linear = _backend.linear
relu = _backend.relu
conv2d = _backend.conv2d
cross_entropy_loss = _backend.cross_entropy_loss
maxpool2d = _backend.maxpool2d
flatten = _backend.flatten
relu2d = _backend.relu2d

# NEW: Multi-channel operations
try:
    conv2d_multi = _backend.conv2d_multi
    relu3d = _backend.relu3d
    maxpool3d = _backend.maxpool3d
    flatten3d = _backend.flatten3d
except AttributeError:
    print("WARNING: Multi-channel operations not found in C++ backend!")
    conv2d_multi = None
    relu3d = None
    maxpool3d = None
    flatten3d = None

try:
    scale_tensor_2d = _backend.scale_tensor_2d
except AttributeError:
    scale_tensor_2d = None