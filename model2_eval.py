import sys
import os
import time

from dataset import ImageFolderDataset
from model_data1 import SimpleCNN
from framework.cpp_backend import Tensor
from framework.loss import CrossEntropyLoss


# =====================================================
# USAGE
# =====================================================

if len(sys.argv) != 3:
    print("Usage:")
    print("  python model1_eval.py <test_dataset_root> <model_weights.txt>")
    sys.exit(1)

TEST_ROOT = sys.argv[1]
WEIGHTS   = sys.argv[2]

print("=" * 60)
print("MODEL-1 EVALUATION")
print("=" * 60)
print("Test Root :", TEST_ROOT)
print("Weights   :", WEIGHTS)
print("=" * 60)


# =====================================================
# LOAD DATASET
# =====================================================

dataset = ImageFolderDataset(TEST_ROOT, preload=False)

print(f"Indexed {len(dataset)} images across {len(dataset.class_map)} classes")
print("Class map:", dataset.class_map)

num_classes = len(dataset.class_map)


# =====================================================
# LOAD MODEL
# =====================================================

model = SimpleCNN(num_classes)
loss_fn = CrossEntropyLoss()


# =====================================================
# LOAD WEIGHTS (model_weights.txt)
# =====================================================

with open(WEIGHTS, "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue

    if line.startswith("conv.W="):
        rows = [eval(r) for r in line.replace("conv.W=", "").split(";") if r]
        model.conv.W.data_mat = rows

    elif line.startswith("conv.b="):
        model.conv.b.data_vec = [float(line.replace("conv.b=", ""))]

    # elif line.startswith("conv2.W="):
    #     rows = [eval(r) for r in line.replace("conv2.W=", "").split(";") if r]
    #     model.conv2.W.data_mat = rows

    # elif line.startswith("conv2.b="):
    #     model.conv2.b.data_vec = [float(line.replace("conv2.b=", ""))]

    elif line.startswith("fc1.W="):
        rows = [eval(r) for r in line.replace("fc1.W=", "").split(";") if r]
        model.fc1.W.data_mat = rows

    elif line.startswith("fc1.b="):
        model.fc1.b.data_vec = [float(v) for v in line.replace("fc1.b=", "").split(",") if v]

    elif line.startswith("fc2.W="):
        rows = [eval(r) for r in line.replace("fc2.W=", "").split(";") if r]
        model.fc2.W.data_mat = rows

    elif line.startswith("fc2.b="):
        model.fc2.b.data_vec = [float(v) for v in line.replace("fc2.b=", "").split(",") if v]

    elif line.startswith("fc3.W="):
        rows = [eval(r) for r in line.replace("fc3.W=", "").split(";") if r]
        model.fc3.W.data_mat = rows

    elif line.startswith("fc3.b="):
        model.fc3.b.data_vec = [float(v) for v in line.replace("fc3.b=", "").split(",") if v]

print("Weights loaded successfully.")


# =====================================================
# EVALUATION
# =====================================================

correct = 0
total = 0
loss_sum = 0.0

t0 = time.time()

for i in range(len(dataset)):

    img = dataset._load_image(dataset.image_paths[i])
    label = dataset.labels[i]

    x = Tensor(img)
    logits = model(x)
    loss = loss_fn(logits, label)

    loss_sum += loss.data

    pred = logits.data_vec.index(max(logits.data_vec))
    if pred == label:
        correct += 1

    total += 1


elapsed = time.time() - t0

acc = correct / total
avg_loss = loss_sum / total


# =====================================================
# RESULTS
# =====================================================

print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)
print(f"Images evaluated : {total}")
print(f"Average Loss     : {avg_loss:.4f}")
print(f"Accuracy         : {acc*100:.2f}%")
print(f"Correct          : {correct}/{total}")
print(f"Eval Time        : {elapsed:.2f}s")
print("=" * 60)