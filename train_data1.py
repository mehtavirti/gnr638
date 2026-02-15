from dataset import ImageFolderDataset
from model_data1 import SimpleCNN, OUTPUT_DIR
from framework.cpp_backend import Tensor
from framework.loss import CrossEntropyLoss
from framework.optim import SGD

import os
import time
import random

# ===== CONFIG =====
DATA_PATH = r"C:\Users\ngc\OneDrive - Indian Institute of Technology Bombay\Documents\gnr638\Assignment_1\Assignment_1\data\data_1"

EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 0.01        
LR_DECAY    = 0.85        
VAL_SPLIT   = 0.2
RANDOM_SEED = 42
PATIENCE    = 8           # stop if val doesn't improve for 8 epochs
SAVE_PATH   = os.path.join(OUTPUT_DIR, "model_weights.txt")
HIST_PATH   = os.path.join(OUTPUT_DIR, "training_history.txt")
# ==================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("FULL TRAINING MODE")
print("=" * 50)
print(f"Epochs        : {EPOCHS}")
print(f"Batch Size    : {BATCH_SIZE}")
print(f"Learning Rate : {LR} (decay x{LR_DECAY} per epoch)")
print(f"Patience      : {PATIENCE} epochs")
print(f"Output dir    : {OUTPUT_DIR}")
print("=" * 50)

dataset = ImageFolderDataset(DATA_PATH, preload=False)

random.seed(RANDOM_SEED)
all_indices = list(range(len(dataset)))
random.shuffle(all_indices)

val_size      = int(len(all_indices) * VAL_SPLIT)
val_indices   = all_indices[:val_size]
train_indices = all_indices[val_size:]

train_paths  = [dataset.image_paths[i] for i in train_indices]
train_labels = [dataset.labels[i]      for i in train_indices]
val_paths    = [dataset.image_paths[i] for i in val_indices]
val_labels   = [dataset.labels[i]      for i in val_indices]

num_classes = len(dataset.class_map)
print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Classes: {num_classes}")

print(f"\nLoading {len(train_paths)} train images into RAM...")
t0 = time.time()
train_images = []
for i, path in enumerate(train_paths):
    train_images.append(dataset._load_image(path))
    if (i + 1) % 5000 == 0:
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed
        eta     = (len(train_paths) - (i + 1)) / rate / 60
        print(f"  {i+1}/{len(train_paths)} | {rate:.0f} img/s | ETA: {eta:.1f} min")
train_load_time = time.time() - t0
print(f"Train in RAM! ({train_load_time/60:.1f} min, {len(train_paths)/train_load_time:.0f} img/s)")

print(f"\nLoading {len(val_paths)} val images into RAM...")
t0 = time.time()
val_images = []
for path in val_paths:
    val_images.append(dataset._load_image(path))
val_load_time = time.time() - t0
print(f"Val in RAM! ({val_load_time:.1f}s)")

print(f"\n[Dataset Loading Time]")
print(f"  Train : {train_load_time:.2f}s  ({train_load_time/60:.1f} min)")
print(f"  Val   : {val_load_time:.2f}s")
print(f"  Total : {train_load_time + val_load_time:.2f}s")

model     = SimpleCNN(num_classes)
loss_fn   = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=LR)

model.print_summary(num_classes)


def evaluate_fast():
    correct = 0
    for i in range(len(val_images)):
        logits = model(Tensor(val_images[i]))
        pred   = logits.data_vec.index(max(logits.data_vec))
        if pred == val_labels[i]:
            correct += 1
    return correct / len(val_images)


def save_model():
    with open(SAVE_PATH, "w") as f:
        f.write(f"num_classes={num_classes}\n")

        f.write("conv.W=")
        for row in model.conv.W.data_mat:
            f.write(f"{row};")
        f.write("\n")
        f.write(f"conv.b={model.conv.b.data_vec[0]}\n")

        f.write("fc1.W=")
        for row in model.fc1.W.data_mat:
            f.write(f"{row};")
        f.write("\n")
        f.write("fc1.b=")
        for v in model.fc1.b.data_vec:
            f.write(f"{v},")
        f.write("\n")

        f.write("fc2.W=")
        for row in model.fc2.W.data_mat:
            f.write(f"{row};")
        f.write("\n")
        f.write("fc2.b=")
        for v in model.fc2.b.data_vec:
            f.write(f"{v},")
        f.write("\n")

        f.write("fc3.W=")
        for row in model.fc3.W.data_mat:
            f.write(f"{row};")
        f.write("\n")
        f.write("fc3.b=")
        for v in model.fc3.b.data_vec:
            f.write(f"{v},")
        f.write("\n")


history        = []
best_val_acc   = 0.0
no_improve_cnt = 0

print("\n" + "=" * 50)
print("Starting Training (images served from RAM)")
print("=" * 50)

overall_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    correct     = 0
    total       = 0
    epoch_loss  = 0.0
    batch_count = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {optimizer.lr:.6f}")

    epoch_indices = list(range(len(train_images)))
    random.shuffle(epoch_indices)

    for batch_start_idx in range(0, len(epoch_indices), BATCH_SIZE):
        batch_idx = epoch_indices[batch_start_idx : batch_start_idx + BATCH_SIZE]

        optimizer.zero_grad()

        for i in batch_idx:
            x      = Tensor(train_images[i])
            logits = model(x)
            loss   = loss_fn(logits, train_labels[i])
            loss.backward()

            epoch_loss += loss.data
            pred = logits.data_vec.index(max(logits.data_vec))
            if pred == train_labels[i]:
                correct += 1
            total += 1

        optimizer.step()
        batch_count += 1

        if batch_count % 50 == 0:
            elapsed = time.time() - epoch_start
            rate    = total / elapsed if elapsed > 0 else float("inf")
            eta_min = ((len(train_images) - total) / rate / 60) if rate > 0 else 0
            print(
                f"  Batch {batch_count}/{len(train_images)//BATCH_SIZE} | "
                f"Train Acc: {correct/total:.3f} | "
                f"Speed: {rate:.0f} img/s | "
                f"ETA: {eta_min:.1f} min"
            )

    train_acc  = correct / total
    train_loss = epoch_loss / total
    epoch_time = time.time() - epoch_start

    val_start = time.time()
    val_acc   = evaluate_fast()
    val_time  = time.time() - val_start

    gap = train_acc - val_acc
    history.append((train_loss, train_acc, val_acc))

    if val_acc > best_val_acc:
        best_val_acc   = val_acc
        no_improve_cnt = 0
        save_model()
        print(f"  *** Best model saved! Val Acc: {val_acc:.4f} ***")
    else:
        no_improve_cnt += 1

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | "
        f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
        f"Gap: {gap:.4f} | "
        f"Time: {epoch_time/60:.1f}min | Val: {val_time:.1f}s"
    )
    print("=" * 50)

    if no_improve_cnt >= PATIENCE:
        print(f"No val improvement for {PATIENCE} epochs — stopping early.")
        break

    optimizer.lr *= LR_DECAY

total_time = time.time() - overall_start
print(f"\nTotal time (incl. disk load) : {(total_time + train_load_time + val_load_time)/60:.1f} min")
print(f"Training time only           : {total_time/60:.1f} min")
print(f"Best Val Accuracy            : {best_val_acc:.4f}")
print(f"Weights saved to             : {SAVE_PATH}")

with open(HIST_PATH, "w") as f:
    f.write("epoch,train_loss,train_acc,val_acc,gap\n")
    for i, (loss, tacc, vacc) in enumerate(history):
        f.write(f"{i+1},{loss:.4f},{tacc:.4f},{vacc:.4f},{tacc-vacc:.4f}\n")
print(f"Training history saved to    : {HIST_PATH}")