"""
inference.py — Deep Learning Visual MCQ Solver  (v4 — constrained extraction)
Usage: python inference.py --test_dir <absolute_path_to_test_dir>
Output: submission.csv saved in the same directory as this script

Key change vs v3:
  - Entire regex parser REMOVED.
  - Instead: after the VLM reasons freely, a second tiny call uses a LogitsProcessor
    that physically masks every token except '1','2','3','4'.
    The model is forced to emit exactly one valid digit — zero parsing needed.
  - Confidence is read from the reasoning text before extraction.
  - LOW confidence → return 5 (protect negative marks).
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
MAX_NEW_TOKENS      = 600
MAX_IMAGE_PIXELS    = 512 * 28 * 28
MIN_VOTES_TO_ANSWER = 1

# ============================================================
# PROMPT  (v3 — stronger fact authority + confidence gate)
# ============================================================
UNIVERSAL_PROMPT = """You are an expert deep learning engineer and mathematician. You can read every character in the image perfectly.
 
The image contains ONE multiple-choice question with exactly 4 options labeled 1/2/3/4 OR A/B/C/D (A=1, B=2, C=3, D=4).
 
════════════════════════════════════════════════
STEP 1 — CHECK FORMULA/FACT TABLE FIRST
Before doing ANY reasoning, look up the relevant formula or fact below.
These facts are AUTHORITATIVE. If your intuition contradicts them, trust this table.
════════════════════════════════════════════════
 
▌PARAMETER COUNTS
• Conv2D params WITH bias: out_ch × (in_ch × k_h × k_w + 1)
  — The "+1" is the bias PER output channel, giving +out_ch total bias params
  — Example: Conv2d(in=3, out=64, k=3×3) → 64 × (3×3×3 + 1) = 64 × 28 = 1792
  — Example: Conv2d(in=3, out=64, k=3×3) WITHOUT bias → 64 × 3×3×3 = 1728
• FC (Linear) params with bias: (in_features × out_features) + out_features
• LSTM total params: 4 × [hidden × (input + hidden) + hidden]
  — LSTM has 4 gates: input, forget, output, cell gate
  — Example: input=10, hidden=20 → 4 × [20×(10+20) + 20] = 4×620 = 2480
• GRU total params: 3 × [hidden × (input + hidden) + hidden]
  ★ GRU has ONLY 3 gates: reset, update, new/candidate gate (NOT 4 like LSTM)
  ★ NEVER use the LSTM formula (×4) for GRU — it MUST be ×3
  — Example: input=10, hidden=20 → 3 × [20×(10+20) + 20] = 3×620 = 1860  (NOT 2480)
  — If you see answer choices 1860 vs 2480 for a GRU question, 1860 is correct
• BatchNorm1d/2d learnable params: 2 × num_features  (γ scale + β shift)
• nn.Embedding params: vocab_size × embedding_dim  (no extra terms)
• ViT patch tokens: (image_size / patch_size)²
  — Example: 224×224 image, 16×16 patches → (224/16)² = 14² = 196 tokens
• Per-head dim in multi-head attention: d_k = d_model / n_heads
  — Example: d_model=512, 8 heads → d_k = 512/8 = 64
 
▌SPATIAL SIZE FORMULAS
• Conv2D output: floor((input + 2×padding − kernel) / stride) + 1
  — Example: input=32, k=5, s=1, p=0 → floor((32+0−5)/1)+1 = 28
  — Example: input=64, k=3, s=2, p=1 → floor((64+2−3)/2)+1 = floor(63/2)+1 = 32
• MaxPool output: floor((input − kernel) / stride) + 1  (assuming p=0)
  — Example: input=32, k=2, s=2 → floor((32−2)/2)+1 = 16
  — Example: input=4,  k=2, s=2 → floor((4−2)/2)+1  = 2
• Effective receptive field of N stacked (k×k, s=1, no-pad) conv layers: N×(k−1)+1
  ★ TWO 3×3 conv layers → 2×(3−1)+1 = 5×5   (NOT 9×9)
  ★ THREE 3×3 conv layers → 3×(3−1)+1 = 7×7
 
▌FLOPS / COMPUTE
• FLOPs for a Linear(M→N) layer on a SINGLE sample: M×N multiplications + M×N additions = 2×M×N
  — If batch size is B: total FLOPs = B × 2 × M × N
  ★ For a batch of B samples through Linear(M,N): answer is B×M×N (multiply-accumulate) or 2×B×M×N (full FLOPs)
  — Pick the option that matches B×M×N; do NOT pick B×N or M×N alone
• FLOPs for Conv2D output spatial H_out×W_out, with kernel k×k, in_ch C_in, out_ch C_out:
  — per output element: 2 × C_in × k × k  (MAC operations)
  — total: 2 × C_in × k² × C_out × H_out × W_out
 
▌LOSS FUNCTIONS
• Cross-entropy for one sample: −log(p_correct)
  — Example: true class prob=0.8 → −log(0.8) ≈ 0.097  (NOT 0.699, NOT 0.155, NOT 0.223)
  — Example: true class prob=0.7 → −log(0.7) ≈ 0.357
  ★ Use natural log (ln), NOT log base 10
• nn.CrossEntropyLoss in PyTorch:
  ★ EXPECTS RAW LOGITS (before any softmax or log-softmax)
  ★ It applies log-softmax internally — do NOT pass softmax output
• Focal loss: DOWN-WEIGHTS easy, well-classified examples; up-weights hard ones
• CTC loss: sequence-to-sequence tasks with UNALIGNED input/output (speech, OCR)
• KL divergence:
  ★ NOT symmetric: KL(P||Q) ≠ KL(Q||P) in general
  ★ BOTH KL(P||Q) and KL(Q||P) are always ≥ 0  — they are NEVER negatives of each other
  ★ They are not equal AND they are not negatives → the correct answer is "asymmetric"
  ★ Equals 0 IFF P = Q exactly
 
▌CLASSIFICATION METRICS — COMPUTE CAREFULLY
• Precision    = TP / (TP + FP)
• Recall       = TP / (TP + FN)
• F1 score     = 2 × Precision × Recall / (Precision + Recall)
  ★ WORKED EXAMPLE: Precision=0.8, Recall=0.6
    F1 = 2×0.8×0.6 / (0.8+0.6) = 0.96/1.4 ≈ 0.686   (NOT 0.70, NOT 0.75, NOT 0.72)
  ★ Another example: Precision=0.9, Recall=0.75
    F1 = 2×0.9×0.75 / (0.9+0.75) = 1.35/1.65 ≈ 0.818
• Accuracy     = (TP + TN) / (TP + TN + FP + FN)
• Recall from counts: Recall = TP / (TP + FN) = (correctly predicted positives) / (total actual positives)
  — Example: 90 correctly predicted out of 120 actual positives → Recall = 90/120 = 0.75
• AUC-ROC of a random classifier = 0.5
 
▌ACTIVATION FUNCTIONS & GRADIENTS
• Sigmoid/Tanh: saturate → vanishing gradients; derivative → 0 for large |z|
• ReLU: gradient = 1 for positive inputs → LEAST prone to vanishing gradients
• Dead ReLU: input always negative → gradient always 0 → weights NEVER update
• Sigmoid derivative at large positive z ≈ 0  (saturated region)
 
▌SOFTMAX & TEMPERATURE
• Softmax formula: softmax(z/T)_i = exp(z_i/T) / Σ exp(z_j/T)
• Higher temperature T → distribution becomes SOFTER / MORE UNIFORM (probabilities closer together)
• Lower  temperature T → distribution becomes SHARPER / MORE PEAKED (one class dominates)
★ T→∞: uniform distribution.  T→0: one-hot (argmax).
• Softmax is invariant to adding any constant to all logits (constant cancels in fraction)
• Scaled dot-product attention: softmax(Q·Kᵀ / √d_k) · V
 
▌MULTI-HEAD ATTENTION — OUTPUT DIMENSIONS
• Each head computes attention over d_k = d_model / n_heads dimensions
• Each head output shape: (seq_len, d_k)
• CONCATENATION of all heads: concat of n_heads × d_k = n_heads × (d_model/n_heads) = d_model
  ★ Concatenated output dimension = d_model  (NOT n_heads × d_model)
  ★ Example: 8 heads, d_model=512 → each head dim=64; concat=8×64=512  (NOT 8×512=4096)
• Final linear projection maps d_model → d_model (does not change dimension)
 
▌WEIGHT INITIALISATION
• Xavier / Glorot → designed for Sigmoid and Tanh (linear-like activations)
• He / Kaiming   → designed for ReLU and its variants (Leaky ReLU, PReLU, ELU)
 
▌REGULARISATION & PRUNING
• L1 (Lasso) → sparsity (drives weights to exactly 0)
• L2 (Ridge / weight decay) → keeps weights small but non-zero
• Dropout: ACTIVE during training, DISABLED during model.eval()
• Gradient clipping → prevents EXPLODING gradients
• Magnitude-based pruning (weight pruning):
  ★ Removes weights with SMALLEST absolute values (closest to zero)
  ★ NEVER removes largest absolute value weights
  — If asked "which weights are pruned first" → answer is smallest magnitude / near-zero weights
 
▌NORMALISATION
• BatchNorm DURING TRAINING: uses CURRENT MINI-BATCH mean and variance
  ★ During training it does NOT use running statistics — it uses the batch itself
• BatchNorm DURING INFERENCE/EVAL: uses accumulated RUNNING mean & running variance
  ★ Running statistics are computed and updated DURING TRAINING (not during eval)
  — Summary: running stats are UPDATED in training, USED in inference
• LayerNorm: normalises across FEATURE dimensions per sample (independent of batch size)
• BatchNorm: normalises across the BATCH dimension
• GroupNorm with 1 group = LayerNorm
• GroupNorm with group_size = 1 = InstanceNorm
• model.eval() does TWO things: (1) switches BatchNorm to running stats AND (2) disables Dropout
  ★ model.eval() does NOT disable gradient computation (that is torch.no_grad())
  ★ model.eval() does NOT affect gradient computation at all
 
▌OPTIMISERS & MOMENTUM
• SGD with momentum: uses EXPONENTIALLY DECAYED weighted sum of past gradients
  ★ Momentum is NOT a simple equal-weight average of past gradients
  ★ Recent gradients are weighted MORE than older gradients (exponential decay)
  — v_t = β × v_{t-1} + (1-β) × g_t   where β is momentum coefficient (typically 0.9)
• RMSProp: adaptive per-parameter rates, NO momentum by default
• Adam: COMBINES momentum + per-parameter adaptive rates  ← "momentum + adaptive" = Adam
• Adagrad: adaptive, monotonically shrinking learning rate
 
▌PYTORCH TRAINING LOOP ORDER
1. optimizer.zero_grad()   ← FIRST — clear old gradients
2. output = model(input)
3. loss = criterion(output, target)
4. loss.backward()
5. optimizer.step()
• torch.no_grad(): disables gradient computation → saves memory; does NOT freeze weights
 
▌model.parameters() AND FROZEN LAYERS
• model.parameters() returns ALL parameters in the model, including frozen ones
  ★ Freezing a layer (requires_grad=False) does NOT remove it from model.parameters()
  ★ To get only trainable params: filter(lambda p: p.requires_grad, model.parameters())
• torch.no_grad(): disables gradient computation → saves memory; does NOT freeze weights
 
▌ARCHITECTURES & TRAINING TRICKS
• ResNet residual block: output = F(x) + x   (identity shortcut = x, not F(x), not W·x)
• U-Net skip connections: recover spatial detail LOST during downsampling
• Depthwise separable conv: reduces parameters and compute (not spatial resolution)
• Stride=2 conv vs max-pool: stride conv is LEARNABLE downsampling
• ConvTranspose2d: UPSAMPLING / increasing spatial resolution
• Sharp minima → POOR generalisation
• Large batch sizes → converge to sharp minima → poorer generalisation
• Gradient accumulation N steps → simulates LARGER effective batch size
• Cosine annealing vs step decay: smooth transitions, avoids abrupt drops
 
▌EFFICIENTNET & NEURAL ARCHITECTURE SCALING
• EfficientNet uses COMPOUND SCALING — scales THREE dimensions simultaneously:
  ★ Width (number of channels)
  ★ Depth (number of layers)
  ★ Resolution (input image size)
  — NOT learning rate, NOT batch size, NOT dropout rate
  — If an option says "width, depth, resolution" → that is correct for EfficientNet
 
▌CONTINUAL LEARNING / TRANSFER LEARNING
• Catastrophic forgetting (catastrophic interference):
  ★ When a neural network FORGETS previously learned tasks upon learning new ones
  ★ It is NOT about slow convergence, NOT about overfitting, NOT about vanishing gradients
  — Occurs because gradient updates for new task overwrite weights learned for old tasks
• Fine-tuning pretrained CNN on small dataset: freeze EARLY conv layers (low-level features) first
• Chest X-ray augmentation: VERTICAL FLIP is inappropriate (breaks anatomical orientation)
 
▌TRANSFORMERS & SELF-SUPERVISED
• Original Transformer positional encoding: FIXED sinusoidal functions of position
• Weight tying: input embedding matrix tied to output projection layer
• GAN generator (original): minimises log(1 − D(G(z)))
• SimCLR NT-Xent: same image different augmentations → similar; different images → far apart
• Knowledge distillation soft targets: carry inter-class similarity information
• VAE reparameterisation trick: makes sampling differentiable for backpropagation
 
════════════════════════════════════════════════
STEP 2 — SOLVE the question carefully
Read ALL 4 options completely. Do NOT assume.
Compute exact values; do NOT round until comparing with options.
════════════════════════════════════════════════

════════════════════════════════════════════════
STEP 3 — VERIFY all 4 options and reject wrong ones explicitly
════════════════════════════════════════════════

════════════════════════════════════════════════
STEP 4 — CONFIDENCE SELF-CHECK
Rate your confidence: HIGH / MEDIUM / LOW
- HIGH: You are certain and the answer exactly matches one option.
- MEDIUM: You are fairly sure but there is some ambiguity.
- LOW: You are guessing or multiple options seem plausible.
Write: CONFIDENCE: HIGH  or  CONFIDENCE: MEDIUM  or  CONFIDENCE: LOW
════════════════════════════════════════════════

════════════════════════════════════════════════
MANDATORY FINAL LINE — must always be last
════════════════════════════════════════════════
FINAL ANSWER: X
Where X is exactly one digit: 1, 2, 3, or 4.
Convert A→1, B→2, C→3, D→4.
If you rated CONFIDENCE: LOW, still write your best guess as FINAL ANSWER.
Do NOT write anything after the FINAL ANSWER line."""


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
# CONFIDENCE READER  (simple — no parsing of answer needed here)
# ============================================================
def extract_confidence(text: str) -> str:
    """Read the CONFIDENCE: tag the model writes, or infer from hedging language."""
    if not text:
        return "LOW"
    t = text.upper()
    m = re.search(r"CONFIDENCE\s*[:\-]?\s*(HIGH|MEDIUM|LOW)", t)
    if m:
        return m.group(1)
    if "HIGH CONFIDENCE" in t or "HIGHLY CONFIDENT" in t:
        return "HIGH"
    if "LOW CONFIDENCE" in t or "NOT CONFIDENT" in t or "UNSURE" in t:
        return "LOW"
    hedge_phrases = [
        "closest option", "closest to", "not sure", "cannot determine",
        "unclear", "ambiguous", "could be either", "i'm not certain",
        "i am not certain", "hard to tell", "it's possible",
    ]
    if any(p in text.lower() for p in hedge_phrases):
        return "LOW"
    return "MEDIUM"


# ============================================================
# CONSTRAINED ANSWER EXTRACTOR
# Forces the model to output exactly ONE token: '1', '2', '3', or '4'.
# No regex. No parsing. Physically impossible to output anything else.
# ============================================================
_VALID_TOKEN_IDS: list | None = None   # populated on first call

def _get_valid_token_ids() -> list[int]:
    """Return the token IDs for '1', '2', '3', '4' in the loaded tokenizer."""
    global _VALID_TOKEN_IDS
    if _VALID_TOKEN_IDS is not None:
        return _VALID_TOKEN_IDS
    tok = processor.tokenizer
    ids = []
    for digit in ["1", "2", "3", "4"]:
        # Most tokenizers have a direct single-token for bare digits
        encoded = tok.encode(digit, add_special_tokens=False)
        # Take only the last token if it encodes as multiple (e.g. BPE prefix space)
        ids.append(encoded[-1])
    _VALID_TOKEN_IDS = ids
    return ids


class _AllowOnlyDigits:
    """LogitsProcessor that sets all logits to -inf except the 4 answer tokens."""
    def __init__(self, allowed_ids: list[int]):
        self.allowed = set(allowed_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.allowed:
            mask[:, tid] = scores[:, tid]
        return mask


# ============================================================
# SEMANTIC ANSWER EXTRACTOR — uses a second model call to understand reasoning
# Much more robust than regex; handles "B. 5130", "option 3", "answer is C", etc.
# ============================================================
def extract_answer_constrained(reasoning_text: str) -> int | None:
    """
    Exhaustive extractor. Handles every known and theoretical output pattern.
    Strategy: clean → regex cascade on tail only → LLM constrained fallback.
    """
    if not reasoning_text:
        return None

    def letter_to_int(c: str) -> int:
        c = c.upper()
        if c.isdigit():
            return int(c)
        return ord(c) - ord('A') + 1

    def clean(text: str) -> str:
        """Strip ALL markdown, normalize whitespace."""
        t = re.sub(r'\*+', ' ', text)       # bold/italic asterisks
        t = re.sub(r'_+', ' ', t)           # underscore bold
        t = re.sub(r'#+\s*', ' ', t)        # headers
        t = re.sub(r'[→⇒✓✗•◆▶]', ' ', t)  # arrows, bullets, symbols
        t = re.sub(r'[–—]', '-', t)         # em/en dash → hyphen
        t = re.sub(r'\s+', ' ', t)          # normalize spaces
        return t.upper().strip()

    def find_option_label(text: str) -> int | None:
        """
        Core regex logic. ONLY matches option LABELS (A/B/C/D or 1/2/3/4).
        Key insight: always use (?=\s|$|\.|,|\)|\]|:) after capture group
        so we never accidentally match a digit INSIDE a value like 5130.
        """
        SEP = r'(?=[\s,\.:\)\]\/]|$)'   # what can follow a label — NOT another digit
        LEAD = r'(?:^|[\s\(\[#\-])'     # what can precede a label

        patterns_letter_first = [
            # === EXPLICIT FINAL ANSWER MARKERS (highest priority) ===
            r'FINAL\s+ANSWER\s*[:\-=]?\s*([A-D])\b',
            r'FINAL\s+ANSWER\s*[:\-=]?\s*([1-4])' + SEP,

            # === "ANSWER IS / ANSWER:" patterns ===
            r'(?:THE\s+)?(?:CORRECT\s+)?(?:BEST\s+)?ANSWER\s+IS\s+(?:OPTION\s+|CHOICE\s+)?([A-D])\b',
            r'(?:THE\s+)?(?:CORRECT\s+)?(?:BEST\s+)?ANSWER\s+IS\s+(?:OPTION\s+|CHOICE\s+)?([1-4])' + SEP,
            r'(?:THE\s+)?(?:CORRECT\s+)?(?:BEST\s+)?ANSWER\s*[:\-=]\s*(?:OPTION\s+|CHOICE\s+)?([A-D])\b',
            r'(?:THE\s+)?(?:CORRECT\s+)?(?:BEST\s+)?ANSWER\s*[:\-=]\s*(?:OPTION\s+|CHOICE\s+)?([1-4])' + SEP,

            # === "CORRECT/RIGHT OPTION/CHOICE/IMPLEMENTATION IS ___" ===
            r'(?:CORRECT|RIGHT)\s+\w+(?:\s+\w+)?\s+IS\s+(?:OPTION\s+|CHOICE\s+)?([A-D])\b',
            r'(?:CORRECT|RIGHT)\s+\w+(?:\s+\w+)?\s+IS\s+(?:OPTION\s+|CHOICE\s+)?([1-4])' + SEP,

            # === "OPTION/CHOICE X IS CORRECT/RIGHT" ===
            r'OPTION\s+([A-D])\s+IS\s+(?:CORRECT|RIGHT|BEST)',
            r'OPTION\s+([1-4])\s+IS\s+(?:CORRECT|RIGHT|BEST)',
            r'CHOICE\s+([A-D])\s+IS\s+(?:CORRECT|RIGHT|BEST)',
            r'CHOICE\s+([1-4])\s+IS\s+(?:CORRECT|RIGHT|BEST)',

            # === "SELECT/CHOOSE/PICK/GO WITH option X" ===
            r'(?:SELECT|CHOOSE|PICK|USE|GO\s+WITH)\s+(?:OPTION\s+|CHOICE\s+)?([A-D])\b',
            r'(?:SELECT|CHOOSE|PICK|USE|GO\s+WITH)\s+(?:OPTION\s+|CHOICE\s+)?([1-4])' + SEP,

            # === "X IS THE CORRECT/RIGHT ANSWER" (letter/digit comes first) ===
            r'\b([A-D])\s+IS\s+(?:THE\s+)?(?:CORRECT|RIGHT|BEST)',
            r'\b([1-4])\s+IS\s+(?:THE\s+)?(?:CORRECT|RIGHT|BEST)' ,

            # === "THEREFORE/THUS/HENCE/SO ... X" ===
            r'(?:THEREFORE|THUS|HENCE|SO)[^.]{0,60}(?:OPTION\s+|CHOICE\s+)?([A-D])\b',
            r'(?:THEREFORE|THUS|HENCE|SO)[^.]{0,60}(?:OPTION\s+|CHOICE\s+)?([1-4])' + SEP,

            # === Parenthesized: "(B)" or "(2)" ===
            r'\(([A-D])\)',
            r'\(([1-4])\)',

            # === "ANS: B" / "ANS = 2" ===
            r'\bANS(?:WER)?\s*[:\-=]\s*([A-D])\b',
            r'\bANS(?:WER)?\s*[:\-=]\s*([1-4])' + SEP,

            # === Bare isolated letter A/B/C/D on its own at end of line ===
            r'(?:^|[\s\(\[])([A-D])(?:[\s\.\)\]]|$)',
        ]

        for pattern in patterns_letter_first:
            matches = list(re.finditer(pattern, text))
            if matches:
                return letter_to_int(matches[-1].group(1))
        return None

    # ── PHASE 1: Search progressively expanding tail of text ──
    # Start with last 3 lines (conclusion), expand if nothing found
    all_lines = [l.strip() for l in reasoning_text.strip().split('\n') if l.strip()]

    for n_lines in [3, 5, 8, len(all_lines)]:
        tail_lines = all_lines[-n_lines:]
        tail_raw   = '\n'.join(tail_lines)
        tail_clean = clean(tail_raw)
        result = find_option_label(tail_clean)
        if result is not None:
            print(f"  [Extractor] Found via regex (tail={n_lines} lines): {result}")
            return result

    # ── PHASE 2: LLM constrained extraction ──
    # Feed only the last ~1500 chars of reasoning, stripped of markdown
    tail_for_llm = clean(reasoning_text[-1500:])

    extraction_prompt = f"""You are an answer extractor. Read the reasoning below and output the option label chosen.

CRITICAL RULES — read carefully:
1. Options are labeled A, B, C, D (same as 1, 2, 3, 4)
2. Options have VALUES like "5130", "28x28", "1792", "0.357" — these are the option CONTENTS, NOT the label
3. "B. 5130" means the student chose label B → you output 2
4. "Option A" means label A → output 1
5. "**Final Answer:** 1" means label 1 → output 1
6. NEVER output a digit you found inside a number like 5130, 1792, 28, etc.
7. Output ONLY one digit: 1, 2, 3, or 4. No words. No punctuation.

Reasoning (markdown stripped):
---
{tail_for_llm}
---

Output the single digit label (1/2/3/4):"""

    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": extraction_prompt}]}]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input], padding=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                logits_processor=[_AllowOnlyDigits(_get_valid_token_ids())],
            )
        trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
        digit_str = processor.decode(trimmed, skip_special_tokens=True).strip()
        if digit_str in {"1", "2", "3", "4"}:
            print(f"  [Extractor] Found via LLM fallback: {digit_str}")
            return int(digit_str)

    except Exception as e:
        print(f"  [Extractor] LLM failed: {e}")

    # ── PHASE 3: Nuclear last resort — isolated letter in last 5 lines ──
    # At this point regex + LLM both failed. Find ANY isolated A/B/C/D.
    tail5 = clean('\n'.join(all_lines[-5:]))
    # Remove all numbers and their surrounding context to avoid value contamination
    tail5_no_numbers = re.sub(r'\b\d+(?:\.\d+)?\b', ' NUM ', tail5)
    isolated = list(re.finditer(r'\b([A-D])\b', tail5_no_numbers))
    if isolated:
        result = letter_to_int(isolated[-1].group(1))
        print(f"  [Extractor] Found via nuclear fallback: {result}")
        return result

    print(f"  [Extractor] TOTAL FAILURE — returning None")
    return None


# ============================================================
# MODEL QUERY
# ============================================================
def query_model(img: Image.Image, prompt: str, temperature: float = 0.1) -> str:
    from qwen_vl_utils import process_vision_info

    # Resize — keep high-res for text legibility
    max_side = 896   # slightly higher than v2's 768 for better text reading
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif max(w, h) < 400:
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
def solve_mcq_image(image_path: str, verbose: bool = True) -> int:
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

    try:
        # ── Stage 1: free-form reasoning
        response = query_model(img, UNIVERSAL_PROMPT, temperature=0.0)
        if verbose:
            print(f"  Response:\n{response}")

        # ── Confidence gate (protects negative marks)
        confidence = extract_confidence(response)
        if verbose:
            print(f"  Confidence: {confidence}")
        if confidence == "LOW":
            if verbose:
                print("  LOW confidence → 5")
            return 5

        # ── Stage 2: constrained extraction — forces exactly one digit
        ans = extract_answer_constrained(response)
        if verbose:
            print(f"  Extracted answer: {ans}")
        if ans is None:
            if verbose:
                print("  Constrained extraction failed → 5")
            return 5

        return ans

    except Exception as e:
        if verbose:
            print(f"  Error: {e} → 5")
        return 5


# ============================================================
# MAIN
# ============================================================
def main():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser(description="MCQ Solver Inference")
    parser.add_argument("--test_dir", required=True,
                        help="Absolute path to the test directory")
    parser.add_argument("--image_path", required=False,
                        help="Run a single image instead of full dataset")
    args = parser.parse_args()
    global processor, model
    if args.image_path:
        print("Running single image mode...\n")

        # Load model (same as below)
        from transformers import (
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
        )

       

        print("Loading processor...")
        processor = Qwen2_5_VLProcessor.from_pretrained(
            HF_MODEL_ID,
            min_pixels=64 * 28 * 28,
            max_pixels=MAX_IMAGE_PIXELS,
        )

        print("Loading model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            HF_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.eval()

        pred = solve_mcq_image(args.image_path, verbose=True)

        print(f"\nFinal Prediction: {pred}")
        return
    test_dir  = args.test_dir
    test_csv  = os.path.join(test_dir, "test.csv")
    image_dir = os.path.join(test_dir, "images")

    from transformers import (
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )

    print("Loading processor...")
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

        pred = solve_mcq_image(img_path, verbose=True)
        predictions.append(pred)

        q_time = time.time() - q_start
        elapsed = time.time() - start_time
        avg = elapsed / (idx + 1)
        eta = avg * (total - idx - 1)

        print(
            f"  [{idx+1}/{total}] {image_name} → {pred} | "
            f"Q: {q_time:.2f}s | Avg: {avg:.2f}s | ETA: {eta/60:.2f}m"
        )

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
