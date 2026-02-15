import sys
import os
import time
import pickle
from PIL import Image

from framework.cpp_backend import Tensor, relu
from framework.loss import CrossEntropyLoss
from framework.conv_multi import Conv2DMulti, MaxPool2DMulti, ReLUMulti, FlattenMulti
from framework.linear import Linear


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

if len(sys.argv) != 3:
    print("Usage:")
    print("  py eval_model2.py <test_parent_directory> <model.pkl>")
    sys.exit(1)

TEST_ROOT = sys.argv[1]
MODEL_PKL = sys.argv[2]


# --------------------------------------------------
# BUILD CLASS MAP FROM SUBFOLDERS
# --------------------------------------------------

class_names = sorted([
    d for d in os.listdir(TEST_ROOT)
    if os.path.isdir(os.path.join(TEST_ROOT, d))
])

class_map = {name: idx for idx, name in enumerate(class_names)}

samples = []

for cls_name in class_names:
    folder = os.path.join(TEST_ROOT, cls_name)
    for img_name in os.listdir(folder):
        if img_name.lower().endswith((".jpg",".png",".jpeg")):
            samples.append(
                (os.path.join(folder, img_name), class_map[cls_name])
            )

print("="*60)
print("MODEL-2 EVALUATION")
print("="*60)
print(f"Test Root : {TEST_ROOT}")
print(f"Images    : {len(samples)}")
print(f"Classes   : {class_map}")
print("="*60)


# --------------------------------------------------
# IMAGE → TENSOR
# --------------------------------------------------

def image_to_tensor(path):
    img = Image.open(path).convert("RGB").resize((32,32))
    pixels = list(img.getdata())

    data = [[[0]*32 for _ in range(32)] for _ in range(3)]

    for i in range(32):
        for j in range(32):
            r,g,b = pixels[i*32+j]
            data[0][i][j] = (r/255-0.5)/0.5
            data[1][i][j] = (g/255-0.5)/0.5
            data[2][i][j] = (b/255-0.5)/0.5

    return Tensor(data)


# --------------------------------------------------
# BUILD MODEL ARCHITECTURE
# --------------------------------------------------

conv1 = Conv2DMulti(3,32,3)
relu1 = ReLUMulti()
pool1 = MaxPool2DMulti(2)

conv2 = Conv2DMulti(32,64,3)
relu2 = ReLUMulti()
pool2 = MaxPool2DMulti(2)

flatten = FlattenMulti()

fc1 = Linear(2304,128)
fc2 = Linear(128,len(class_map))

# --------------------------------------------------
# LOAD PICKLE CHECKPOINT
# --------------------------------------------------

with open(MODEL_PKL, "rb") as f:
    ckpt = pickle.load(f)

conv1.W.data_4d = ckpt["conv1_W"]
conv1.b.data_vec = ckpt["conv1_b"]

conv2.W.data_4d = ckpt["conv2_W"]
conv2.b.data_vec = ckpt["conv2_b"]

fc1.W.data_mat = ckpt["fc1_W"]
fc1.b.data_vec = ckpt["fc1_b"]

fc2.W.data_mat = ckpt["fc2_W"]
fc2.b.data_vec = ckpt["fc2_b"]

print("Model loaded successfully.")


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

loss_fn = CrossEntropyLoss()

correct = 0
total_loss = 0.0

start = time.time()

for path, label in samples:

    x = image_to_tensor(path)

    x = conv1(x)
    x = relu1(x)
    x = pool1(x)
    x = conv2(x)
    x = relu2(x)
    x = pool2(x)
    x = flatten(x)
    x = fc1(x)
    x = relu(x)
    x = fc2(x)

    pred = x.data_vec.index(max(x.data_vec))

    if pred == label:
        correct += 1

    loss = loss_fn(x, label)
    total_loss += loss.data

total = len(samples)
accuracy = correct / total
avg_loss = total_loss / total

print("\nRESULTS")
print("="*60)
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Loss     : {avg_loss:.4f}")
print(f"Correct  : {correct}/{total}")
print(f"Time     : {time.time()-start:.2f}s")
print("="*60)
