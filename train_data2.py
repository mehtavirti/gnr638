import os, random, time, pickle, math
from PIL import Image

from framework.cpp_backend import Tensor, relu
from framework.loss import CrossEntropyLoss
from framework.optim import SGD
from framework.conv_multi import Conv2DMulti, MaxPool2DMulti, ReLUMulti, FlattenMulti
from framework.linear import Linear


DATA_PATH = r"C:\Users\ngc\OneDrive - Indian Institute of Technology Bombay\Documents\gnr638\Assignment_1\Assignment_1\data\data_2"
SAVE_PATH = r"C:\Users\ngc\OneDrive - Indian Institute of Technology Bombay\Documents\gnr638\Assignment_1\Assignment_1\checkpoints"


def argmax(vec):
    """Return index of maximum value in a flat list."""
    best_idx = 0
    best_val = vec[0]
    for i in range(1, len(vec)):
        if vec[i] > best_val:
            best_val = vec[i]
            best_idx = i
    return best_idx


def image_to_3d_list(img):
    """
    Convert a PIL RGB image (already resized to 32x32) into a
    3-D Python list  [C][H][W].
    """
    width, height = img.size          # 32, 32
    pixels = list(img.getdata())      # list of (R, G, B) tuples, row-major

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Allocate [3][32][32]
    data = [[[0.0] * width for _ in range(height)] for _ in range(3)]

    for row in range(height):
        for col in range(width):
            pixel = pixels[row * width + col]   # (R, G, B)  0-255
            for c in range(3):
                v = pixel[c] / 255.0            # [0, 1]
                v = (v - mean[c]) / std[c]      # ImageNet normalisation
                data[c][row][col] = v

    return data


def gauss_random(mean, std):
    """Box-Muller transform for Gaussian random numbers (no numpy)."""
    while True:
        u1 = random.random()
        u2 = random.random()
        if u1 > 0.0:
            break
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z


# ============================================================
# DATASET
# ============================================================

class ImageFolderDataset:
    def __init__(self, samples):
        self.samples = samples

    def load_image(self, path):
        img = Image.open(path).convert("RGB").resize((32, 32))
        data_3d = image_to_3d_list(img)          # pure Python list [3][32][32]
        return Tensor(data_3d)                   # 3D constructor in C++ backend

    def __getitem__(self, idx):
        return self.load_image(self.samples[idx][0]), self.samples[idx][1]

    def __len__(self):
        return len(self.samples)


def load_samples(root):
    samples = []
    for idx, cls in enumerate(sorted(os.listdir(root))):
        cp = os.path.join(root, cls)
        if os.path.isdir(cp):
            for img in os.listdir(cp):
                if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                    samples.append((os.path.join(cp, img), idx))
    return samples


# ============================================================
# START
# ============================================================

print("=" * 70)
print("OPTIMIZED RGB CNN")
print("=" * 70)

os.makedirs(SAVE_PATH, exist_ok=True)

print("\n[1/4] Loading data...")
t_load_start = time.time()
all_samples = load_samples(DATA_PATH)
t_load_end = time.time()
print(f"Dataset loading time: {t_load_end - t_load_start:.2f} seconds")

random.shuffle(all_samples)
split = int(0.9 * len(all_samples))
train_set = ImageFolderDataset(all_samples[:split])
val_set   = ImageFolderDataset(all_samples[split:])

print(f"Train: {len(train_set)}, Val: {len(val_set)}")


# ============================================================
# MODEL
# ============================================================

print("\n[2/4] Creating layers...")

conv1 = Conv2DMulti(3, 32, 3)
relu1 = ReLUMulti()
pool1 = MaxPool2DMulti(2)

conv2 = Conv2DMulti(32, 64, 3)
relu2 = ReLUMulti()
pool2 = MaxPool2DMulti(2)

flatten = FlattenMulti()

print("Layers created successfully.")


# ============================================================
# FC INIT
# ============================================================

print("\n[3/4] Initializing FC layers...")

x_init, _ = train_set[0]

print("Running shape inference forward pass...")
x = conv1(x_init)
x = relu1(x)
x = pool1(x)
x = conv2(x)
x = relu2(x)
x = pool2(x)
x = flatten(x)

flat_size = len(x.data_vec)
print(f"Flatten size: {flat_size}")

fc1 = Linear(flat_size, 128)
std1 = math.sqrt(2.0 / flat_size)
fc1.W.data_mat = [
    [gauss_random(0.0, std1) for _ in range(128)]
    for _ in range(flat_size)
]
fc1.b.data_vec = [0.0] * 128

fc2 = Linear(128, 100)
std2 = math.sqrt(2.0 / 128)
fc2.W.data_mat = [
    [gauss_random(0.0, std2) for _ in range(100)]
    for _ in range(128)
]
fc2.b.data_vec = [0.0] * 100

print("FC layers initialized.")


def count_params_and_macs(in_ch, H, W):
    # --- conv1: 3 -> 32 filters, 3x3 ---
    p_conv1   = 32 * 3 * 3 * 3 + 32
    mac_conv1 = 32 * 3 * 3 * 3 * (H-2) * (W-2)
    H1, W1    = (H-2)//2, (W-2)//2

    # --- conv2: 32 -> 64 filters, 3x3 ---
    p_conv2   = 64 * 32 * 3 * 3 + 64
    mac_conv2 = 64 * 32 * 3 * 3 * (H1-2) * (W1-2)
    H2, W2    = (H1-2)//2, (W1-2)//2

    flat      = 64 * H2 * W2

    # --- fc1: flat -> 128 ---
    p_fc1     = flat * 128 + 128
    mac_fc1   = flat * 128

    # --- fc2: 128 -> 100 ---
    p_fc2     = 128 * 100 + 100
    mac_fc2   = 128 * 100

    total_params = p_conv1 + p_conv2 + p_fc1 + p_fc2
    total_macs   = mac_conv1 + mac_conv2 + mac_fc1 + mac_fc2
    total_flops  = 2 * total_macs

    print("\n" + "="*50)
    print("MODEL COMPLEXITY")
    print("="*50)
    print(f"  Trainable parameters : {total_params:,}")
    print(f"  MACs per forward pass: {total_macs:,}")
    print(f"  FLOPs per forward pass: {total_flops:,}")
    print("="*50 + "\n")

count_params_and_macs(3, 32, 32)


params = [
    conv1.W, conv1.b,
    conv2.W, conv2.b,
    fc1.W, fc1.b,
    fc2.W, fc2.b
]

criterion = CrossEntropyLoss()
optimizer = SGD(params, lr=0.001)


# ============================================================
# TRAINING
# ============================================================

print("\n[4/4] Training...")
print("=" * 70)

EPOCHS      = 20
PRINT_EVERY = 200

best       = 0
best_epoch = 0
start_time = time.time()

for epoch in range(EPOCHS):

    epoch_start = time.time()
    random.shuffle(train_set.samples)

    loss_sum = 0.0
    correct  = 0

    print(f"\n--- Epoch {epoch+1}/{EPOCHS} started ---")

    for i in range(len(train_set)):

        x, y = train_set[i]

        # Forward
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

        optimizer.zero_grad()
        loss = criterion(x, y)

        if argmax(x.data_vec) == y:
            correct += 1

        loss.backward()
        optimizer.step()

        loss_sum += loss.data

        if (i + 1) % PRINT_EVERY == 0:
            elapsed = time.time() - epoch_start
            rate    = (i + 1) / elapsed
            eta     = (len(train_set) - (i + 1)) / rate

            print(
                f"E{epoch+1:02d} "
                f"[{i+1:5d}/{len(train_set)}] "
                f"Loss:{loss_sum/(i+1):.3f} "
                f"Acc:{100*correct/(i+1):5.2f}% "
                f"ETA:{int(eta/60):02d}:{int(eta%60):02d}",
                end="\r"
            )

    # ---------------- VALIDATION ----------------
    vc          = 0
    val_samples = min(200, len(val_set))

    for i in range(val_samples):
        x, y = val_set[i]

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

        if argmax(x.data_vec) == y:
            vc += 1

    va         = vc / val_samples
    epoch_time = time.time() - epoch_start

    print(f"\nE{epoch+1:02d}: "
          f"Train {100*correct/len(train_set):5.2f}% | "
          f"Val {100*va:5.2f}% | "
          f"Loss {loss_sum/len(train_set):.4f} | "
          f"Time {epoch_time/60:.1f} min")

    if va > best:
        best       = va
        best_epoch = epoch + 1

        model_name = f"best_model_E{epoch+1}_VA{100*va:.2f}.pkl"

        with open(os.path.join(SAVE_PATH, model_name), 'wb') as f:
            checkpoint = {
                "conv1_W": conv1.W.data_4d,
                "conv1_b": conv1.b.data_vec,
                "conv2_W": conv2.W.data_4d,
                "conv2_b": conv2.b.data_vec,
                "fc1_W":   fc1.W.data_mat,
                "fc1_b":   fc1.b.data_vec,
                "fc2_W":   fc2.W.data_mat,
                "fc2_b":   fc2.b.data_vec,
            }
            pickle.dump(checkpoint, f)

        print(f"    ✓ Saved {model_name}")

    if epoch == 10:
        optimizer.lr = 0.0005
        print("    Learning rate reduced to 0.0005")

    if epoch == 15:
        optimizer.lr = 0.0001
        print("    Learning rate reduced to 0.0001")


print("\nTraining Complete.")
print(f"Best Validation Accuracy: {100*best:.2f}% (Epoch {best_epoch})")
print(f"Total training time: {(time.time()-start_time)/60:.1f} min")