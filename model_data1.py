from framework.conv2d import Conv2D
from framework.maxpool import MaxPool2D
from framework.flatten import Flatten
from framework.linear import Linear
from framework.cpp_backend import relu, relu2d

# ===== OUTPUT CONFIG =====
OUTPUT_DIR = "outputs"
# =========================


class SimpleCNN:
    def __init__(self, num_classes):
        self.conv = Conv2D(3)
        self.pool = MaxPool2D()
        self.flat = Flatten()
        self.fc1  = Linear(15 * 15, 256)
        self.fc2  = Linear(256, 128)
        self.fc3  = Linear(128, num_classes)

    def __call__(self, x):
        x = self.conv(x)
        x = relu2d(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.fc3(x)
        return x

    def parameters(self):
        return [
            self.conv.W, self.conv.b,
            self.fc1.W,  self.fc1.b,
            self.fc2.W,  self.fc2.b,
            self.fc3.W,  self.fc3.b,
        ]

    def print_summary(self, num_classes):
        conv_w       = 3 * 3
        conv_b       = 1
        fc1_w        = 225 * 256
        fc1_b        = 256
        fc2_w        = 256 * 128
        fc2_b        = 128
        fc3_w        = 128 * num_classes
        fc3_b        = num_classes
        total_params = conv_w + conv_b + fc1_w + fc1_b + fc2_w + fc2_b + fc3_w + fc3_b

        # Conv2D: 3x3 kernel, 1 input channel (grayscale 32x32), output 30x30
        conv_macs   = 30 * 30 * 3 * 3 * 1
        fc1_macs    = 225 * 256
        fc2_macs    = 256 * 128
        fc3_macs    = 128 * num_classes
        total_macs  = conv_macs + fc1_macs + fc2_macs + fc3_macs
        total_flops = 2 * total_macs

        print("\n" + "=" * 50)
        print("MODEL COMPLEXITY")
        print("=" * 50)
        print(f"  Trainable parameters  : {total_params:,}")
        print(f"  MACs per forward pass : {total_macs:,}")
        print(f"  FLOPs per forward pass: {total_flops:,}")
        print("=" * 50)
        print()