import sys
import os
import time
import pickle
import warnings
from PIL import Image

from framework.cpp_backend import Tensor, relu
from framework.loss import CrossEntropyLoss
from framework.conv_multi import Conv2DMulti, MaxPool2DMulti, ReLUMulti, FlattenMulti
from framework.linear import Linear

warnings.filterwarnings("ignore", category=DeprecationWarning)

if len(sys.argv) != 3:
    print("Usage:")
    print("  py eval_model2.py <test_parent_directory> <model.pkl>")
    sys.exit(1)

TEST_ROOT = sys.argv[1]
MODEL_PKL = sys.argv[2]

class_names = sorted([
    d for d in os.listdir(TEST_ROOT)
    if os.path.isdir(os.path.join(TEST_ROOT, d))
])
class_map = {name: idx for idx, name in enumerate(class_names)}

samples = []
for cls_name in class_names:
    folder = os.path.join(TEST_ROOT, cls_name)
    for img_name in os.listdir(folder):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            samples.append((os.path.join(folder, img_name), class_map[cls_name]))

print("="*60)
print("MODEL-2 EVALUATION")
print("="*60)
print(f"Test Root : {TEST_ROOT}")
print(f"Images    : {len(samples)}")
print(f"Classes   : {class_map}")
print("="*60)

with open(MODEL_PKL, "rb") as f:
    ckpt = pickle.load(f)

# Read sizes directly from weights
conv1_out = len(ckpt["conv1_W"])           # 16
conv1_in  = len(ckpt["conv1_W"][0])        # 3
conv1_k   = len(ckpt["conv1_W"][0][0])     # kernel size

conv2_out = len(ckpt["conv2_W"])           # 32
conv2_in  = len(ckpt["conv2_W"][0])        # 16
conv2_k   = len(ckpt["conv2_W"][0][0])     # kernel size

fc1_in    = len(ckpt["fc1_W"])             # 512
fc1_out   = len(ckpt["fc1_W"][0])          # 128

fc2_in    = len(ckpt["fc2_W"])             # 128
fc2_out   = len(ckpt["fc2_W"][0])          # 100

print(f"Architecture from pkl:")
print(f"  conv1 : {conv1_in}->{conv1_out}, kernel={conv1_k}")
print(f"  conv2 : {conv2_in}->{conv2_out}, kernel={conv2_k}")
print(f"  fc1   : {fc1_in}->{fc1_out}")
print(f"  fc2   : {fc2_in}->{fc2_out}")
print("="*60)

import math
target = fc1_in // conv2_out  
side = int(math.isqrt(target))
IMG_SIZE = (side*2 + 2)*2 + 2
print(f"  Inferred image size: {IMG_SIZE}x{IMG_SIZE}")
print("="*60)


def image_to_tensor(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    pixels = list(img.getdata())

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    data = [[[0.0]*IMG_SIZE for _ in range(IMG_SIZE)] for _ in range(3)]
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            r, g, b = pixels[i*IMG_SIZE+j]
            data[0][i][j] = (r/255.0 - mean[0]) / std[0]
            data[1][i][j] = (g/255.0 - mean[1]) / std[1]
            data[2][i][j] = (b/255.0 - mean[2]) / std[2]

    return Tensor(data)


# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------

conv1 = Conv2DMulti(conv1_in, conv1_out, conv1_k)
relu1 = ReLUMulti()
pool1 = MaxPool2DMulti(2)

conv2 = Conv2DMulti(conv2_in, conv2_out, conv2_k)
relu2 = ReLUMulti()
pool2 = MaxPool2DMulti(2)

flatten = FlattenMulti()

fc1 = Linear(fc1_in, fc1_out)
fc2 = Linear(fc2_in, fc2_out)

conv1.W.data_4d  = ckpt["conv1_W"]
conv1.b.data_vec = ckpt["conv1_b"]
conv2.W.data_4d  = ckpt["conv2_W"]
conv2.b.data_vec = ckpt["conv2_b"]
fc1.W.data_mat   = ckpt["fc1_W"]
fc1.b.data_vec   = ckpt["fc1_b"]
fc2.W.data_mat   = ckpt["fc2_W"]
fc2.b.data_vec   = ckpt["fc2_b"]

print("Model loaded successfully.")
print(f"Evaluating {len(samples)} images...\n")

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

loss_fn    = CrossEntropyLoss()
correct    = 0
total_loss = 0.0
start      = time.time()

for idx, (path, label) in enumerate(samples):

    x = image_to_tensor(path)
    x = conv1(x);  x = relu1(x);  x = pool1(x)
    x = conv2(x);  x = relu2(x);  x = pool2(x)
    x = flatten(x)
    x = fc1(x);    x = relu(x)
    x = fc2(x)

    pred = x.data_vec.index(max(x.data_vec))
    if pred == label:
        correct += 1

    loss = loss_fn(x, label)
    total_loss += loss.data

    print(f"  [{idx+1}/{len(samples)}] pred={pred} label={label} {'✓' if pred==label else '✗'}")

total    = len(samples)
accuracy = correct / total
avg_loss = total_loss / total

print("\nRESULTS")
print("="*60)
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Loss     : {avg_loss:.4f}")
print(f"Correct  : {correct}/{total}")
print(f"Time     : {time.time()-start:.2f}s")
print("="*60)