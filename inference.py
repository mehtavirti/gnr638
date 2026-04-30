"""
inference.py — Deep Learning Visual MCQ Solver
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
HF_MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"  # loaded from HF cache
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, "submission.csv")

# ============================================================
# INFERENCE SETTINGS
# ============================================================
N_VOTES             = 1      # set to 1 to safely stay under 1-hr limit
LLM_TEMP            = 0.5
MAX_NEW_TOKENS      = 1024
MAX_IMAGE_PIXELS    = 1280 * 28 * 28
MIN_VOTES_TO_ANSWER = 1

# ============================================================
# PROMPT
# ============================================================
UNIVERSAL_PROMPT = """You are an expert in deep learning, machine learning, mathematics, and PyTorch.

You will be shown an image with a multiple-choice question and exactly 4 options.

STEP 0 — TOPIC HINT: Look at the very top of the image. If there is a line starting with "Ques:" read it carefully and use it to guide your reasoning.

STEP 1 — QUESTION TYPE: Identify what this question involves — it may be a mix of calculation, code, architecture, and/or conceptual reasoning. List all that apply.

STEP 2 — SOLVE: Work out the correct answer using whatever reasoning is needed.
Show all working clearly. Use the options only if the question requires comparing implementations or code.

STEP 3 — EVALUATE: For each option, state whether it is correct or incorrect and why.
- Option 1/A:
- Option 2/B:
- Option 3/C:
- Option 4/D:

STEP 4 — SANITY CHECK:
- Re-state in one sentence what your derived answer IS.
- Re-read each option and confirm which one best matches that meaning — not just the first one that sounds related.
- For "why/primary reason" questions: are you picking the ROOT cause or a side effect?
- For PyTorch API questions: are you confusing what a function takes as INPUT vs what it outputs?
- For training curve questions: re-read the exact direction of each loss (↑ or ↓) per option.
- If this check changes your mind, update your answer now before STEP 5.

STEP 5 — CONFIDENCE: HIGH / MEDIUM / LOW

STEP 6 — FINAL ANSWER: X
Options may be labeled 1/2/3/4 OR A/B/C/D — look at the image to check.
Always convert to a digit: A=1, B=2, C=3, D=4.
Final answer must be a single digit: 1, 2, 3, or 4 — nothing else.

Key facts:
- Conv2D output: floor((input + 2*padding - kernel) / stride) + 1
- Conv2D params: (k_h * k_w * in_ch + 1) * out_ch
- FC params with bias: (in * out) + out
- LSTM params: 4 * (hidden * (input + hidden) + hidden)
- BatchNorm params: 2 * num_features
- Attention: softmax(Q*K^T / sqrt(d_k)) * V
- nn.Sequential executes layers strictly top to bottom in the order listed
- Activation functions (ReLU, Sigmoid, Tanh etc.) always come AFTER the linear layer they activate
- To implement h = ReLU(Wx + b): the ONLY correct PyTorch order is nn.Linear THEN nn.ReLU() — always
- nn.Linear(a,b) followed by nn.ReLU() means: apply linear first, then ReLU — this is CORRECT
- Never say ReLU after Linear is wrong — it is always the right order
- Trace each equation LEFT TO RIGHT: innermost operation maps to first layer in Sequential
- h1 = ReLU(W1x+b1) → nn.Linear(in,h1), nn.ReLU()
- h2 = ReLU(W2h1+b2) → nn.Linear(h1,h2), nn.ReLU()
- y = W3h2+b3 → nn.Linear(h2,out)  [no activation]
- Full sequence: Linear→ReLU→Linear→ReLU→Linear = CORRECT for a 2-hidden-layer MLP with ReLU
- Dropout: ON at train, OFF at eval
- Vanishing gradient: sigmoid/tanh suffer, ReLU does not
- L1 → sparsity, L2 → small weights
- Batch norm inference: running mean/variance from training
- Sharp minima = poor generalization
- All positive Hessian eigenvalues = local min, mixed = saddle point
- When evaluating options, always simplify and substitute expressions before declaring an option incorrect
- Two expressions that look different may be algebraically identical — verify by substituting definitions
- If your derived answer and an option look different, try expanding/simplifying both sides before rejecting it
IMPORTANT: If the options in the image are labeled with NUMBERS (1, 2, 3, 4),
your FINAL ANSWER must be that number directly — never use A/B/C/D in that case.
If options are labeled with LETTERS (A, B, C, D), convert: A=1, B=2, C=3, D=4.
"""


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
# ANSWER PARSER
# ============================================================
def parse_answer(text: str):
    if not text or not text.strip():
        return None

    letter_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if "final answer" in line.lower():
            digit_match  = re.search(r"(?<![A-Za-z])([1-4])(?![A-Za-z0-9])", line)
            letter_match = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", line, re.IGNORECASE)
            if digit_match:
                return digit_match.group(1)
            if letter_match:
                return letter_map[letter_match.group(1).upper()]

    for line in reversed(lines):
        digit_match = re.search(r"(?<![A-Za-z])([1-4])(?![A-Za-z0-9])", line)
        if digit_match:
            return digit_match.group(1)

    matches = re.findall(r"\b([1-4])\b", text)
    if matches:
        return matches[-1]
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

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            do_sample=(temperature > 0.05),
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


# ============================================================
# SOLVE ONE IMAGE
# ============================================================
def solve_mcq_image(image_path: str, use_voting: bool = False, verbose: bool = True) -> int:
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
        min_pixels=256 * 28 * 28,
        max_pixels=MAX_IMAGE_PIXELS,
    )
    print("Processor loaded!")

    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
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

        elapsed = time.time() - start_time
        avg     = elapsed / (idx + 1)
        eta     = avg * (total - idx - 1)
        print(f"  [{idx+1}/{total}] {image_name} → {pred} | "
              f"{elapsed/60:.1f}m elapsed | ETA {eta/60:.1f}m")

    # ---- Save submission.csv in script's directory ----
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
