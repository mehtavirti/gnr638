import sys
import time
import os

from dataset import ImageFolderDataset
from model1 import SimpleCNN
from framework.cpp_backend import Tensor
from framework.loss import CrossEntropyLoss


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

if len(sys.argv) != 3:
    print("Usage:")
    print("  py eval_model1.py <test_parent_directory> <model_weights.txt>")
    sys.exit(1)

TEST_ROOT   = sys.argv[1]
WEIGHTS_TXT = sys.argv[2]


# --------------------------------------------------
# LOAD DATASET (subfolder name = label)
# --------------------------------------------------

dataset = ImageFolderDataset(TEST_ROOT)
num_classes = len(dataset.class_map)

print("="*60)
print("MODEL-1 EVALUATION")
print("="*60)
print(f"Test Root : {TEST_ROOT}")
print(f"Images    : {len(dataset)}")
print(f"Classes   : {dataset.class_map}")
print("="*60)


# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------

model = SimpleCNN(num_classes)


# --------------------------------------------------
# LOAD WEIGHTS
# --------------------------------------------------

with open(WEIGHTS_TXT) as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue

    if line.startswith("conv1.W="):
        model.conv1.W.data_mat = [eval(r) for r in line.replace("conv1.W=","").split(";") if r]

    elif line.startswith("conv1.b="):
        model.conv1.b.data_vec = [float(line.split("=")[1])]

    elif line.startswith("conv2.W="):
        model.conv2.W.data_mat = [eval(r) for r in line.replace("conv2.W=","").split(";") if r]

    elif line.startswith("conv2.b="):
        model.conv2.b.data_vec = [float(line.split("=")[1])]

    elif line.startswith("fc1.W="):
        model.fc1.W.data_mat = [eval(r) for r in line.replace("fc1.W=","").split(";") if r]

    elif line.startswith("fc1.b="):
        model.fc1.b.data_vec = [float(x) for x in line.replace("fc1.b=","").split(",") if x]

    elif line.startswith("fc2.W="):
        model.fc2.W.data_mat = [eval(r) for r in line.replace("fc2.W=","").split(";") if r]

    elif line.startswith("fc2.b="):
        model.fc2.b.data_vec = [float(x) for x in line.replace("fc2.b=","").split(",") if x]

    elif line.startswith("fc3.W="):
        model.fc3.W.data_mat = [eval(r) for r in line.replace("fc3.W=","").split(";") if r]

    elif line.startswith("fc3.b="):
        model.fc3.b.data_vec = [float(x) for x in line.replace("fc3.b=","").split(",") if x]

print("Weights loaded successfully.")


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

loss_fn = CrossEntropyLoss()

correct = 0
total_loss = 0.0

start = time.time()

for img, label in zip(dataset.images, dataset.labels):

    x = Tensor(img)
    logits = model(x)

    pred = logits.data_vec.index(max(logits.data_vec))

    if pred == label:
        correct += 1

    loss = loss_fn(logits, label)
    total_loss += loss.data

total = len(dataset)
accuracy = correct / total
avg_loss = total_loss / total

print("\nRESULTS")
print("="*60)
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Loss     : {avg_loss:.4f}")
print(f"Correct  : {correct}/{total}")
print(f"Time     : {time.time()-start:.2f}s")
print("="*60)
