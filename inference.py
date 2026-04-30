"""
inference.py — Deep Learning Visual MCQ Solver  (patched v2)
Usage: python inference.py --test_dir <absolute_path_to_test_dir>
Output: submission.csv saved in the same directory as this script
"""

import argparse
import os
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

warnings.filterwarnings("ignore")

# ============================================================
# PATHS — model weights cached locally by setup.bash
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
HF_MODEL_ID  = "/content/gnr638/model_3b"
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, "submission.csv")

# ============================================================
# INFERENCE SETTINGS
# ============================================================
N_VOTES             = 1      # set to 1 to safely stay under 1-hr limit
LLM_TEMP            = 0.5
MAX_NEW_TOKENS      = 512
MAX_IMAGE_PIXELS    = 512 * 28 * 28
MIN_VOTES_TO_ANSWER = 1

# ============================================================
# PROMPT  (v2 — expanded with conceptual anchors + formula fixes)
# ============================================================
UNIVERSAL_PROMPT = """You are an expert in deep learning, machine learning, mathematics, and PyTorch. You have perfect vision and can read all text, code, and equations in images clearly.

You will be shown an image containing a multiple-choice question with exactly 4 options labeled either 1/2/3/4 or A/B/C/D.

INSTRUCTIONS:
1. READ the entire image carefully — question, all 4 options completely.
2. SOLVE the problem step by step showing all work.
3. IDENTIFY which single option is correct and why.
4. VERIFY by checking all other options are wrong.

══════════════════════════════════════════════
KEY FORMULAS (use these exactly — do not guess)
══════════════════════════════════════════════
• Conv2D output size: floor((input + 2*padding - kernel) / stride) + 1
• Conv2D params (with bias): (k_h * k_w * in_ch + 1) * out_ch
  — Example: Conv2d(3, 64, kernel=3) → (3*3*3 + 1)*64 = 1792   ← bias adds OUT_CH, not 1
• FC (Linear) params with bias: (in * out) + out
• LSTM total params: 4 * (hidden * (input + hidden) + hidden)
• BatchNorm1d params: 2 * num_features  (gamma + beta, no bias counted separately)
• nn.Embedding params: vocab_size * embedding_dim   ← no +1 for padding unless padding_idx set
• Effective receptive field of N stacked (k×k, stride=1) conv layers: N*(k-1)+1
  — Two 3×3 conv layers → 2*(3-1)+1 = 5×5
• Scaled dot-product attention: softmax(Q * K^T / sqrt(d_k)) * V   ← always divide by sqrt(d_k)
• Per-head dim in multi-head attention: d_k = d_model / n_heads
• Cross-entropy loss for one sample: -log(p_correct_class)  ← only the true class term
• Softmax temperature T: higher T → softer/more uniform; lower T → sharper/more peaked
• Softmax numerical stability trick: subtract max(z) → valid because softmax is invariant
  to adding a constant to all logits (the constant cancels in numerator and denominator)

══════════════════════════════════════════════
NEURAL NETWORK LAYER ORDER (nn.Sequential)
══════════════════════════════════════════════
• h1 = ReLU(W1*x + b1)  →  nn.Linear(in_dim, h1), nn.ReLU()
• h2 = ReLU(W2*h1 + b2) →  nn.Linear(h1, h2),    nn.ReLU()
• y  = W3*h2 + b3        →  nn.Linear(h2, out_dim)  — NO activation at output

══════════════════════════════════════════════
CONCEPTUAL FACTS (memorise — these are tested)
══════════════════════════════════════════════
WEIGHT INITIALISATION:
• Xavier / Glorot init  → designed for Sigmoid and Tanh activations
• He / Kaiming init     → designed for ReLU and its variants (Leaky ReLU, PReLU)

ACTIVATION FUNCTIONS:
• Vanishing gradient problem: affects Sigmoid and Tanh (saturate at extremes)
• ReLU is least prone to vanishing gradients (gradient is 1 for positive inputs)
• Dead ReLU: input always negative → gradient always 0 → weights never update
• Sigmoid derivative for large positive z → approaches 0 (saturated)

REGULARISATION:
• L1 (Lasso)  → encourages weight sparsity (drives weights to exactly 0)
• L2 (Ridge / weight decay) → keeps weights small but non-zero
• Dropout: ACTIVE during training, DISABLED during eval (model.eval())
• Gradient clipping → prevents EXPLODING gradients (not vanishing)

BATCH / LAYER NORMALISATION:
• BatchNorm at inference: uses running mean & running variance (accumulated during training)
• LayerNorm normalises across features per sample
• BatchNorm normalises across the batch dimension
• GroupNorm with 1 group = LayerNorm; GroupNorm with group_size=1 = InstanceNorm
• model.eval() switches BatchNorm to running stats AND disables dropout

OPTIMISERS:
• SGD: no adaptive rates, optional momentum
• RMSProp: adaptive per-parameter rates, no momentum
• Adam: combines momentum + per-parameter adaptive learning rates  ← correct answer for "momentum + adaptive"
• Adagrad: adaptive, accumulates all past gradients (learning rate shrinks monotonically)

LOSS FUNCTIONS:
• nn.CrossEntropyLoss in PyTorch expects RAW LOGITS (not softmax/log-softmax output)
• Focal loss: down-weights EASY, well-classified examples; up-weights hard misclassified ones
• CTC loss: used for sequence-to-sequence tasks with UNALIGNED input/output (e.g. OCR, ASR)
• KL divergence: NOT symmetric; KL(P||Q) ≠ KL(Q||P); always ≥ 0; = 0 iff P=Q exactly

TRAINING DYNAMICS:
• Sharp minima → poor generalisation
• Large batch sizes → tend to converge to sharp minima → poorer generalisation
• Gradient accumulation over N steps → simulates LARGER effective batch size
• Cosine annealing advantage over step decay → smooth transitions, avoids abrupt drops

TRANSFORMERS & ATTENTION:
• Original Transformer positional encoding: fixed sinusoidal functions of position
• Weight tying in language models: input EMBEDDING matrix tied to output PROJECTION layer
• GAN generator (original objective): minimises log(1 - D(G(z)))
  Practical alternative (non-saturating): maximise log(D(G(z))) = minimise -log(D(G(z)))

ARCHITECTURES:
• U-Net skip connections: recover spatial detail lost during downsampling
• ResNet residual block output: F(x) + x  — the identity shortcut adds x, not F(x)
• Depthwise separable convolution: reduces parameters and compute (not spatial resolution)
• Stride=2 convolution vs max-pool: stride conv is LEARNABLE downsampling
• ConvTranspose2d (transposed conv): used for UPSAMPLING / increasing spatial resolution
• ViT patch tokens for 224×224 image with 16×16 patches: (224/16)^2 = 196 tokens

TRANSFER LEARNING:
• Fine-tuning a pretrained CNN on small dataset: freeze EARLY conv layers first (low-level features)
  — Train the final classification head which is randomly initialised
• Data augmentation for chest X-rays: vertical flip (upside-down) is INAPPROPRIATE
  — Flipping vertically breaks anatomical orientation (heart position, lung shape)

SELF-SUPERVISED / DISTILLATION:
• SimCLR NT-Xent loss: same image under different augmentations → similar; different images → far apart
• Knowledge distillation soft targets: carry inter-class similarity information
• VAE reparameterisation trick: makes sampling differentiable for backpropagation
• torch.no_grad(): disables gradient computation → reduces memory; does NOT freeze weights permanently

PYTORCH TRAINING LOOP ORDER:
1. optimizer.zero_grad()   ← BEFORE loss.backward() (clear old gradients)
2. output = model(input)
3. loss = criterion(output, target)
4. loss.backward()
5. optimizer.step()

══════════════════════════════════════════════
CRITICAL RULES
══════════════════════════════════════════════
- Read EVERY option fully before answering — do not assume options are identical
- If two options look similar, find the exact difference
- Never pick an answer based on partial reading
- Algebraically simplify expressions before marking them wrong
- AUC-ROC of a random classifier = 0.5

FINAL ANSWER FORMAT (mandatory, always the very last line of response):
FINAL ANSWER: X
Where X is a single digit 1, 2, 3, or 4.
If options are A/B/C/D convert: A=1, B=2, C=3, D=4.
Never write anything after the FINAL ANSWER line."""


# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def is_dark_background(img: Image.Image) -> bool:
    arr = np.array(img.convert("L"))
    return arr.mean() < 127


def auto_invert(img: Image.Image) -> Image.Image:
    if is_dark_background(img):
        img = ImageOps.invert(img.convert("RGB"))
    return img


def auto_crop_whitespace(img: Image.Image, padding: int = 10) -> Image.Image:
    arr    = np.array(img.convert("L"))
    bg_val = arr[0, 0]
    mask   = np.abs(arr.astype(int) - int(bg_val)) > 15
    rows   = np.any(mask, axis=1)
    cols   = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = arr.shape
    return img.crop((
        max(0, cmin - padding), max(0, rmin - padding),
        min(w, cmax + padding), min(h, rmax + padding)
    ))


def deskew(img: Image.Image) -> Image.Image:
    try:
        arr = np.array(img.convert("L"))
        _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5 or abs(angle) > 45:
            return img
        h, w = arr.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(
            np.array(img), M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(rotated)
    except Exception:
        return img


def super_resolve(img: Image.Image, target_width: int = 1200) -> Image.Image:
    w, h = img.size
    if w >= 800:
        return img
    scale = target_width / w
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def preprocess_image(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img = auto_invert(img)
    img = deskew(img)
    img = auto_crop_whitespace(img)
    img = super_resolve(img)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img


# ============================================================
# ANSWER PARSER  (v2 — hardened for letter-only answers)
# ============================================================
def parse_answer(text: str):
    if not text or not text.strip():
        return None

    letter_map = {"A": "1", "B": "2", "C": "3", "D": "4"}

    # Strip "closest option" hedge sentences before parsing
    # e.g. "The closest option to 577 is 576." can mislead the parser
    text_clean = re.sub(
        r"[Tt]he closest (?:option|answer)[^.\n]*\.",
        "",
        text
    )
    text_clean = re.sub(
        r"[Tt]he (?:nearest|most similar) (?:option|answer)[^.\n]*\.",
        "",
        text_clean
    )

    lines = text_clean.strip().split("\n")
    lines_orig = text.strip().split("\n")

    # Pass 1: explicit FINAL ANSWER line (bottom up)
    for line in reversed(lines):
        if "final answer" in line.lower():
            cleaned = re.sub(r"[*_`]", "", line)
            m = re.search(r"(?<![A-Za-z])([1-4])(?![A-Za-z0-9])", cleaned)
            if m: return m.group(1)
            m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", cleaned, re.IGNORECASE)
            if m: return letter_map[m.group(1).upper()]

    # Pass 2: "answer is X" / "correct answer is X"
    conclusion_pattern = re.compile(
        r"(?:answer|correct option|correct answer)\s+is\s*[:\-]?\s*\**([1-4ABCD])\**",
        re.IGNORECASE
    )
    for line in reversed(lines):
        m = conclusion_pattern.search(line)
        if m:
            val = m.group(1).upper()
            return letter_map.get(val, val)

    # Pass 3: bare letter/number on its own line e.g. "A" "B." "**C**"
    for line in reversed(lines):
        cleaned = re.sub(r"[*_`.\s]", "", line.strip())
        if re.match(r"^[ABCD]$", cleaned, re.IGNORECASE):
            return letter_map[cleaned.upper()]
        if re.match(r"^[1-4]$", cleaned):
            return cleaned

    # Pass 4: "A. 0.80" or "C. 64" — letter dot then value
    for line in reversed(lines):
        m = re.match(r"^\s*([ABCD])\.\s+\S", line.strip(), re.IGNORECASE)
        if m: return letter_map[m.group(1).upper()]
        m = re.match(r"^\s*([1-4])\.\s+\S", line.strip())
        if m: return m.group(1)

    # Pass 5: Option X is correct
    for line in lines_orig:
        low = line.lower()
        if "incorrect" in low: continue
        m = re.search(r"[Oo]ption\s*([1-4ABCD])", line)
        if m and "correct" in low:
            val = m.group(1).upper()
            return letter_map.get(val, val)

    # Pass 6: last digit fallback
    matches = re.findall(r"\b([1-4])\b", text_clean)
    if matches: return matches[-1]
    return None


def extract_confidence(text: str) -> str:
    if not text:
        return "LOW"
    t = text.upper()
    if re.search(r"CONFIDENCE\s*[:\-]?\s*HIGH", t):   return "HIGH"
    if re.search(r"CONFIDENCE\s*[:\-]?\s*MEDIUM", t): return "MEDIUM"
    if re.search(r"CONFIDENCE\s*[:\-]?\s*LOW", t):    return "LOW"
    if "HIGH CONFIDENCE" in t or "HIGHLY CONFIDENT" in t: return "HIGH"
    if "LOW CONFIDENCE"  in t or "NOT CONFIDENT" in t or "UNSURE" in t: return "LOW"
    return "MEDIUM"


# ============================================================
# SELF-CONTRADICTION CHECK
# ============================================================
def extract_derived_sequence(response: str):
    all_code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
    if not all_code_blocks:
        return None
    best_layers = []
    for block in all_code_blocks:
        layers = re.findall(r"nn\.\w+", block)
        if len(layers) > len(best_layers):
            best_layers = layers
    return best_layers if best_layers else None


def check_self_contradiction(response: str, final_answer: int):
    derived_layers = extract_derived_sequence(response)
    if not derived_layers:
        return None
    letter_map    = {"A": 1, "B": 2, "C": 3, "D": 4}
    options_found = re.findall(
        r"[Oo]ption\s*([1234ABCD])[:/\s].*?```python(.*?)```",
        response, re.DOTALL
    )
    for opt_label, opt_code in options_found:
        opt_layers = re.findall(r"nn\.\w+", opt_code)
        if opt_layers == derived_layers:
            opt_num = letter_map.get(opt_label.upper())
            if opt_num is None:
                try:
                    opt_num = int(opt_label)
                except Exception:
                    continue
            if opt_num != final_answer:
                print(f"  Self-contradiction detected! Correcting {final_answer} → {opt_num}")
                return opt_num
    return None


# ============================================================
# MODEL QUERY
# ============================================================
def query_model(img: Image.Image, prompt: str, temperature: float = 0.1) -> str:
    from qwen_vl_utils import process_vision_info  # imported here to keep globals clean

    # Resize — 3B has headroom, use higher res for text legibility
    max_side = 768
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif max(w, h) < 400:
        # Upscale tiny images so model can read small text
        scale = 400 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text":  prompt},
        ],
    }]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS)
        if temperature > 0.05:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


# ============================================================
# SOLVE ONE IMAGE
# ============================================================
def solve_mcq_image(image_path: str, use_voting: bool = False, verbose: bool = True) -> int:
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    fname = os.path.basename(image_path)
    if verbose:
        print(f"\nProcessing: {fname}")

    try:
        img = preprocess_image(image_path)
    except Exception as e:
        if verbose:
            print(f"  Preprocess failed ({e}), using raw image")
        img = Image.open(image_path).convert("RGB")

    if not use_voting:
        try:
            response = query_model(img, UNIVERSAL_PROMPT, temperature=0.0)
            if verbose:
                print(f"  Response:\n{response}")
            ans = parse_answer(response)
            if ans is None:
                if verbose:
                    print("  Could not parse answer → 5")
                return 5
            corrected = check_self_contradiction(response, int(ans))
            return corrected if corrected is not None else int(ans)
        except Exception as e:
            if verbose:
                print(f"  Error: {e} → 5")
            return 5

    # Voting mode
    votes, confidences = [], []
    for run_idx in range(N_VOTES):
        temp = 0.0 if run_idx == 0 else LLM_TEMP
        try:
            response = query_model(img, UNIVERSAL_PROMPT, temperature=temp)
            ans = parse_answer(response)
            if ans is None:
                continue
            corrected = check_self_contradiction(response, int(ans))
            final = corrected if corrected is not None else int(ans)
            votes.append(final)
            confidences.append(extract_confidence(response))
        except Exception as e:
            if verbose:
                print(f"  Run {run_idx} error: {e}")

    if not votes:
        return 5

    most_common, count = Counter(votes).most_common(1)[0]
    if N_VOTES == 3 and len(set(votes)) == 3:
        return 5
    if count < MIN_VOTES_TO_ANSWER:
        return 5

    if verbose:
        print(f"  Final: {most_common} ({count}/{len(votes)} votes)")
    return most_common


# ============================================================
# MAIN
# ============================================================
def main():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"  # no internet during inference
    parser = argparse.ArgumentParser(description="MCQ Solver Inference")
    parser.add_argument("--test_dir", required=True,
                        help="Absolute path to the test directory")
    args = parser.parse_args()

    test_dir  = args.test_dir
    test_csv  = os.path.join(test_dir, "test.csv")
    image_dir = os.path.join(test_dir, "images")

    # ---- Load model (offline, from local cache set up by setup.bash) ----
    from transformers import (
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )

    print("Loading processor...")
    global processor, model
    processor = Qwen2_5_VLProcessor.from_pretrained(
        HF_MODEL_ID,
        min_pixels=64 * 28 * 28,
        max_pixels=MAX_IMAGE_PIXELS,
    )
    print("Processor loaded!")

    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    max_memory = {0: "12GiB", "cpu": "20GiB"}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
    )
    model.eval()
    print("Model loaded!")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used  = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — {used:.2f}/{total:.1f} GB VRAM")

    # ---- Load test data ----
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        print(f"Loaded test.csv: {test_df.shape}, columns: {test_df.columns.tolist()}")
        if "image_id" not in test_df.columns:
            test_df["image_id"] = test_df["image_name"]
    else:
        img_files = sorted(
            list(Path(image_dir).glob("*.png")) +
            list(Path(image_dir).glob("*.jpg"))
        )
        test_df = pd.DataFrame({
            "image_id":   [f.stem for f in img_files],
            "image_name": [f.stem for f in img_files],
        })
        print(f"No test.csv found. Built template for {len(test_df)} images.")

    total = len(test_df)
    print(f"\nTotal questions: {total}")

    predictions = []
    start_time  = time.time()

    for idx, row in test_df.iterrows():
        q_start = time.time()

        image_name = str(row["image_name"])

        candidates = [
            os.path.join(image_dir, f"{image_name}.png"),
            os.path.join(image_dir, f"{image_name}.jpg"),
            os.path.join(image_dir, image_name),
            os.path.join(image_dir, f"{image_name}.PNG"),
            os.path.join(image_dir, f"{image_name}.jpeg"),
        ]
        img_path = next((p for p in candidates if os.path.exists(p)), None)

        if img_path is None:
            print(f"  [{idx+1}/{total}] NOT FOUND: {image_name} → 5")
            predictions.append(5)
            continue

        pred = solve_mcq_image(img_path, use_voting=(N_VOTES > 1), verbose=True)
        predictions.append(pred)

        q_time = time.time() - q_start
        elapsed = time.time() - start_time
        avg = elapsed / (idx + 1)
        eta = avg * (total - idx - 1)

        print(
            f"  [{idx+1}/{total}] {image_name} → {pred} | "
            f"Q: {q_time:.2f}s | Avg: {avg:.2f}s | ETA: {eta/60:.2f}m"
        )

    # ---- Save submission.csv ----
    submission_df = pd.DataFrame({
        "id":         test_df["image_id"].values,
        "image_name": test_df["image_name"].values,
        "option":     predictions,
    })
    submission_df.to_csv(OUTPUT_PATH, index=False)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done in {total_time/60:.1f} minutes")
    print(f"Answered : {len([p for p in predictions if p != 5])}")
    print(f"Skipped  : {predictions.count(5)}")
    print(f"Distribution: {dict(Counter(predictions))}")
    print(f"Saved to : {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
