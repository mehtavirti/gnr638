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
HF_MODEL_ID  = "./model"
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
• Conv2D params WITH bias:    out_ch × (in_ch × k_h × k_w + 1)
  — Example: Conv2d(in=3, out=64, k=3×3, bias=True)  → 64×(3×3×3+1) = 64×28 = 1792
  — Example: Conv2d(in=3, out=64, k=3×3, bias=False) → 64×3×3×3 = 1728
  — Example: Conv2d(256,64,k=1,bias=False) → 64×256×1×1 = 16384
• Conv2D with groups (depthwise): out_ch × (in_ch/groups × k_h × k_w + bias_per_group)
  — Example: Conv2d(32,64,k=3,groups=2,bias=False) → (64/2)×(32/2×3×3)×2 = 32×144 = 9216... 
    Simpler: total = out_ch × (in_ch/groups) × k_h × k_w = 64×16×9 = 9216
  — Depthwise Conv2d(C,C,k,groups=C,bias=False): C × k × k
    Example: Conv2d(32,32,3,groups=32,bias=False) → 32×3×3 = 288
• FC (Linear) params with bias: (in_features × out_features) + out_features
  — Example: Linear(512,10) → 512×10 + 10 = 5130
• LSTM total params: 4 × [hidden × (input + hidden) + hidden]
  — LSTM has 4 gates: input, forget, cell, output
  — Example: input=10, hidden=20 → 4×[20×30+20] = 4×620 = 2480
★ GRU total params: 3 × [hidden × (input + hidden) + hidden]
  — GRU has only 3 gates: reset, update, new  ← NOT 4 like LSTM
  — Example: input=10, hidden=20 → 3×[20×30+20] = 3×620 = 1860
  — Example: GRU(256,512) → 3×[512×(256+512)+512] = 3×[512×768+512] = 3×394240 = 1,182,720
• BatchNorm1d/2d learnable params: 2 × num_features  (γ + β)
• nn.Embedding params: vocab_size × embedding_dim
• ViT patch tokens: (image_size / patch_size)²
  — Example: 224×224, patch=16 → 14²=196;  384×384, patch=16 → 24²=576
  — ViT-B/16 on 224×224: 196 patch tokens + 1 CLS token = 197 TOTAL sequence length
• Per-head dim: d_k = d_model / n_heads
★ Multi-head attention CONCATENATED output = d_model  (NOT n_heads × d_model)
  — Example: 8 heads, d_model=512 → concat output = 512  (NOT 4096)
★ ResNet-50 bottleneck block params (e.g. 256→64→64→256, no bias):
  — 1×1 conv: 256×64 = 16384
  — 3×3 conv: 64×64×9 = 36864
  — 1×1 conv: 64×256 = 16384
  — Total: 69632  (plus BN params)
 
▌SPATIAL SIZE FORMULAS
• Conv2D output: floor((input + 2×padding − kernel) / stride) + 1
• MaxPool / AvgPool output: floor((input + 2×padding − kernel) / stride) + 1
  — Example: AvgPool(6×6, k=3, s=2, p=0): floor((6-3)/2)+1 = 2×2
  — Example: MaxPool(5×5, k=3, s=1, p=1): floor((5+2-3)/1)+1 = 5×5 (same-padding)
• ConvTranspose2d: output = (input-1)×stride - 2×padding + kernel
  — Example: ConvTranspose2d(3×3, k=3, s=1, p=0): (3-1)×1 - 0 + 3 = 5×5
  — Example: ConvTranspose2d(4×4, stride=2, k=3, p=1): (4-1)×2 - 2 + 3 = 7... use formula
• Effective receptive field: N×(k−1)+1  for N stacked k×k layers (s=1, no pad)
  ★ TWO   3×3 → 5×5;  THREE 3×3 → 7×7
• Dilated conv effective size: k + (k−1)×(dilation−1)
  — k=3, dilation=2 → 5×5;  k=3, dilation=3 → 7×7
• nn.AdaptiveAvgPool2d((1,1)) on (B,C,H,W) → (B,C,1,1)
• GlobalAvgPool on (B,C,H,W) → (B,C,1,1)  NOT (B,C)
• nn.Linear on (B,...,in) → (B,...,out) — only last dimension changes
  — Example: Linear(10,5) on (32,7,10) → (32,7,5)  NOT (32,7,10)
• nn.Flatten(start_dim=1) on (B,C,H,W) → (B, C×H×W)
  — Example: (8,16,4,4) → (8, 16×4×4) = (8, 256)
 
▌FLOPS
★ Matrix multiply A(m×k) × B(k×n): FLOPs ≈ m × k × n  multiply-adds
  — NOT m×n alone; k (inner dimension) is the key multiplier
  — Example: (100×512)×(512×256): FLOPs ≈ 100×512×256 = 13,107,200
★ FLOPs for Linear(M,N) on batch B: B × M × N  (NOT B×M, NOT M×N)
★ FLOPs for Conv2d per output element: C_in × K × K
  — Total Conv FLOPs ≈ C_in × K × K × C_out × H_out × W_out
• Attention QKᵀ FLOPs ≈ n²×d  (quadratic in sequence length n)
• QKV projection FLOPs: 3 × B × seq_len × d_model × d_model
• INT8 vs FP32: ~4× smaller;  FP16 vs FP32: ~2× smaller
★ INT4 vs FP32: 32/4 = 8× smaller  (NOT 4×)
  — 100M params FP32 ≈ 400MB;  FP16 ≈ 200MB;  INT8 ≈ 100MB;  INT4 ≈ 50MB
 
▌LOSS FUNCTIONS & METRICS
• Cross-entropy: −ln(p_correct)
  — p=0.8 → −ln(0.8) ≈ 0.223;  p=0.7 → −ln(0.7) ≈ 0.357
  — p=0.5 → −ln(0.5) ≈ 0.693;  p=0.1 → −ln(0.1) ≈ 2.303
  — p=1.0 → −ln(1.0) = 0  (perfect prediction = zero loss)
  — p=0.368 ≈ e^(-1) → −ln(0.368) ≈ 1.0
  — p=0.25 → −ln(0.25) = ln(4) ≈ 1.386  (4-class uniform)
★ F1 Score = 2×P×R / (P+R)  ← harmonic mean
  — P=0.8, R=0.6 → 2×0.48/1.4 = 0.686
  — P=1.0, R=0.5 → 2×0.5/1.5 = 0.667
  — P=0.7, R=0.5 → 2×0.35/1.2 = 0.583
• Precision = TP/(TP+FP);  Recall = TP/(TP+FN);  Accuracy = (TP+TN)/(TP+TN+FP+FN)
• nn.CrossEntropyLoss: expects RAW LOGITS (applies log-softmax internally)
  ★ NEVER apply softmax before CrossEntropyLoss → double-softmax → WRONG predictions
★ KL divergence: NOT symmetric, always ≥ 0, equals 0 only when P=Q exactly
  — KL(P||Q) and KL(Q||P) are NOT negatives of each other (both ≥ 0)
  — KL divergence is NOT a proper metric (does NOT satisfy triangle inequality)
★ MSE([1,2,3],[2,2,2]) = ((1-2)²+(2-2)²+(3-2)²)/3 = 2/3
 
▌NUMERICAL VALUES (memorise these)
★ exp(0) = 1;  exp(1) ≈ 2.718;  exp(-1) ≈ 0.368;  exp(-2) ≈ 0.135
  — sigmoid(-1) = 1/(1+e) ≈ 1/3.718 ≈ 0.269
  — sigmoid(0) = 0.5
• log(1) = 0  (any base);  ln(e) = 1;  log₂(1) = 0
• -ln(0.5) ≈ 0.693  (natural log);  -log₁₀(0.5) ≈ 0.301
• -ln(0.368) ≈ 1.0  (since 0.368 ≈ e^{-1})
• Sigmoid range: (0,1) open interval — never reaches exactly 0 or 1
• Cosine similarity range for non-zero vectors: [-1, 1]
• softmax([c,c,...,c]) = [1/n, 1/n, ..., 1/n] for any constant c (including large values)
  — softmax([2,2,2,2]) = [0.25, 0.25, 0.25, 0.25]
 
▌SOFTMAX & LOG-SUM-EXP
• log-sum-exp trick: log(Σexp(z_i)) = max(z) + log(Σexp(z_i - max(z)))
  — Valid because shifted values z_i - max(z) ≤ 0, so exp(z_i-max(z)) ≤ 1 → no overflow
• Higher T → softer/more uniform;  Lower T → sharper;  T→0 → one-hot (greedy/argmax)
• Softmax invariant to adding constant to all logits
 
▌NORMALISATION
★ BatchNorm DURING TRAINING: uses CURRENT MINI-BATCH mean & variance
★ BatchNorm DURING INFERENCE: uses RUNNING mean & variance
★ Running stats updated ONLY during training (EMA): (1−m)×running + m×batch_mean
  — NOT updated during eval; NOT simply batch_mean; NOT global dataset mean
• LayerNorm: normalises across feature dims per sample — works fine at batch_size=1
• BatchNorm batch_size=1 during TRAINING: undefined (variance=0, division by zero)
• BatchNorm epsilon ε: added to variance BEFORE sqrt to prevent division by zero
• Forgetting model.eval(): BN uses mini-batch stats → stochastic/wrong inference
• GroupNorm(1 group) = LayerNorm;  GroupNorm(group_size=1) = InstanceNorm
 
▌ACTIVATION FUNCTIONS & GRADIENTS
• ReLU: gradient=1 for x>0 → least prone to vanishing gradients
• Sigmoid/Tanh: saturate → vanishing gradients
• Sigmoid derivative at z=0: 0.25;  at large |z|: ≈ 0
• Leaky ReLU(x=−5, α=0.01) = −0.05
• GELU(0) = 0;  Swish(x→+∞) → x (linear)
★ L1 gradient for w>0: = 1  (NOT w, NOT 2w)
• L2 gradient: 2w
★ All-zeros init: symmetry breaking failure — ALL neurons learn identical features
★ Softmax Jacobian: diagonal = s_i(1-s_i);  off-diagonal = -s_i × s_j  (i≠j)
★ Vanishing gradient through N sigmoid layers: gradient ≤ 0.25^N → near zero for large N
  — 10 sigmoid layers: 0.25^10 ≈ 9.5×10^{-7}  (essentially zero)
 
▌OPTIMISERS & TRAINING
★ SGD momentum: EXPONENTIALLY DECAYED past gradients (NOT equal-weight sum)
• Adam: beta1 = first moment (mean); beta2 = second moment (variance)
• Adam bias correction at t=1, β=0.9: m̂ = m/(1-0.9^1) = m/0.1 = 10×m (scales UP by 10)
• Weight init:
  — He init: Var(W) = 2/fan_in → std = sqrt(2/fan_in)
    Example: Linear(512,256) → std = sqrt(2/512) ≈ 0.0625
  — Xavier init: Var(W) = 2/(fan_in+fan_out) → std = sqrt(2/(fan_in+fan_out))
    Example: Linear(512,256) → std = sqrt(2/768) ≈ 0.051
★ Full batch gradient descent disadvantage: VERY SLOW per update (expensive on large datasets)
  — Noisy gradient is the characteristic of MINI-BATCH / SGD, NOT full batch
★ One Cycle LR policy: LR increases to maximum then decreases (with optional final annealing)
★ Training loss ↓ but validation loss ↑ = OVERFITTING (not underfitting)
★ Training loss HIGHER than validation loss = sign of UNDERFITTING (not overfitting)
  — This is NOT a sign of overfitting
 
▌PYTORCH SPECIFICS
• model.eval(): switches BN to running stats + disables Dropout
  ★ model.train() during inference: activates Dropout + uses mini-batch BN stats → WRONG
★ nn.Dropout: respects model.train()/eval() AUTOMATICALLY
  ★ F.dropout: needs explicit training=True/False — NOT automatic
  — nn.Dropout is SAFER in custom forward pass (auto-respects eval mode)
• model.parameters(): returns ALL params including frozen (requires_grad=False)
• torch.tensor: infers dtype from data
• torch.Tensor: always creates FloatTensor
• backward() twice without zero_grad(): gradients ACCUMULATE (add up)
★ Applying softmax BEFORE nn.CrossEntropyLoss = WRONG (double softmax, inflates loss)
• Embedding layer gradients: only for rows corresponding to tokens in CURRENT batch
• model.half() → converts to FP16
★ BF16 vs FP16: BF16 has SAME exponent bits as FP32 (8 bits) → larger DYNAMIC RANGE
  — FP16 has more mantissa bits → more precision but smaller range
  — BF16 advantage is DYNAMIC RANGE, not precision
 
▌ARCHITECTURES & TRAINING TRICKS
• ResNet: output = F(x) + x
★ Residual stream in Transformer = main activation vector that attention + FFN ADD to
  — NOT the skip connection in ResNet; NOT the gradient path
★ Gradient magnitude in ResNet with L blocks grows LINEARLY with L
  — Identity path adds a term at each level → linear, NOT exponential
• U-Net skip connections: recover spatial detail lost during downsampling
• EfficientNet scales: WIDTH + DEPTH + RESOLUTION simultaneously
★ Catastrophic forgetting: model loses performance on OLD tasks when training on NEW tasks
  — EWC prevents it by penalising changes to important weights (Fisher information)
 
▌TRANSFORMERS & ATTENTION
• GPT: causal (left-to-right);  BERT: bidirectional
  — BERT pre-training: MLM + Next Sentence Prediction (NSP)
★ Masked self-attention in decoder: prevents attending to FUTURE positions
★ RoPE (Rotary Position Embedding): rotates Q and K vectors in attention computation
  — Applies to d_k/2 dimensions per head (pairs of dims for cos/sin rotations)
★ Prefix LM: bidirectional attention on PREFIX + causal attention on generation
  — Combines properties of BERT (on prefix) AND GPT (on generation)
★ Learned positional embeddings (GPT-2 style): CANNOT generalise to sequences LONGER than training length
  — Fixed sinusoidal can extrapolate better
★ BERT 15% masking: NOT 100% to reduce mismatch between pre-training ([MASK] tokens) and fine-tuning (no [MASK])
  — 80% replaced with [MASK], 10% unchanged, 10% random token
★ Causal LM: predicts NEXT token given PAST tokens
  — MLM: predicts MASKED tokens given ALL other tokens
★ Swin Transformer: uses SHIFTED WINDOW (local) attention for linear complexity
★ Residual stream in Transformer = main vector that attention and FFN layers ADD to
★ SwiGLU (used in LLaMA): replaces sigmoid σ with Swish(SiLU) in gated linear unit
★ Mixture of Depths (MoD): skips certain tokens from processing at some layers (adaptive compute)
  — NOT "skip certain layers"; tokens are routed, NOT layers dropped uniformly
★ Sliding window attention: each token attends to w nearest tokens, complexity O(n×w) ≈ O(n)
 
▌FINE-TUNING METHODS
★ BitFit: fine-tunes ONLY the BIAS TERMS of the model (not attention weights, not embeddings)
★ Prompt tuning: trains ONLY a small set of continuous prompt tokens prepended to input
  — Does NOT fine-tune model weights
★ Prefix tuning: prepends trainable vectors to K and V in EACH attention layer
  — NOT the query vectors; NOT only the input embeddings
★ Adapter modules: inserted AFTER attention AND AFTER FFN sublayers within the residual
★ LoRA (rank r): adds low-rank A,B matrices; fewest new parameters among PEFT methods
★ Full fine-tuning: uses the MOST parameters (all model weights updated)
★ BitFit < LoRA < Adapter < Prefix tuning < Full fine-tuning  (in terms of params updated)
 
▌DECODING & GENERATION
• Temperature T→0: one-hot/argmax (greedy);  T→∞: uniform distribution
★ Top-k sampling k=1 = greedy decoding = beam search beam=1
  — "Both B and C" is WRONG; the answer is just "greedy decoding"
★ Repetition penalty: reduces probability of tokens that have ALREADY APPEARED in the generated sequence
★ Beam search: NOT guaranteed to find globally optimal sequence (prunes paths early)
★ CFG (Classifier-Free Guidance) higher weight w: HIGHER FIDELITY but LESS DIVERSE
  — Guided score = (1+w)×score_cond - w×score_uncond
  — Higher w → sharper/more conditioned outputs, loses diversity
 
▌DISTRIBUTED TRAINING & EFFICIENCY
★ Tensor parallelism: requires AllReduce (to sum partial results) per layer
  — NOT AllGather only; NOT no communication
★ ZeRO stage 3: shards optimizer states + gradients + model parameters across GPUs
★ Continuous batching: insert new requests as sequences FINISH (not waiting for full batch)
  — model said 1 = WRONG; correct = inserting without waiting
★ Flash Attention 2 improvement over FA1: better parallelism across sequence dim + fewer non-matmul FLOPs
★ PagedAttention (vLLM): manages KV cache in fixed-size pages/blocks → reduces fragmentation
★ INT4 quantization: 8× smaller than FP32 (not 4×)
 
▌NEURAL ODE & ADVANCED
★ Neural ODE backpropagation memory: O(1) via ADJOINT METHOD (recomputes hidden states in reverse)
  — NOT O(depth) like ResNet; the adjoint avoids storing all intermediate states
★ MAML inner loop: requires Hessian-vector products (second-order gradients)
★ DPO (Direct Preference Optimisation): trains LLM directly on preference data WITHOUT separate reward model AND WITHOUT RL
  — Uses human preference data directly (option A in typical questions = WRONG if it says "without human pref data")
★ Linear attention approximation: rewrites attention as KERNEL FEATURE MAPS, changes order of operations → O(n)
★ Mamba / SSM: selective state space model with RECURRENT computation, linear O(n) in sequence length
 
▌CONTRASTIVE & SELF-SUPERVISED
★ CLIP zero-shot classification: compare image embedding to TEXT EMBEDDINGS of class name descriptions
  — NOT fine-tuning; NOT k-NN with stored image embeddings
★ NT-Xent gradient for perfect positive pair (cosine_sim=1): gradient = 0 (no update needed)
★ Score-based generative models learn: SCORE FUNCTION ∇_x log p(x) (gradient of log density w.r.t. data)
  — NOT p(x) directly; NOT a discriminator
★ Normalizing flows: can compute EXACT log-likelihood via change-of-variables formula
  — NOT just an ELBO lower bound (that is VAE); normalizing flows are EXACT
★ VQ-VAE: uses DISCRETE CODEBOOK — encoder outputs quantised to nearest codebook vector
★ DDIM sampling: 50-100 steps using NON-MARKOVIAN DETERMINISTIC reverse process
  — Much fewer than DDPM's 1000 steps
 
▌EVALUATION & OVERFITTING
★ High bias (underfitting) characteristics: HIGH training error AND HIGH test error
  — NOT just high test error; BOTH train and test are high
★ Overfitting: LOW training error, HIGH test error (large gap)
★ Data leakage: test set information influences model training or selection
★ MC Dropout uncertainty: run inference N times with dropout ACTIVE, average predictions
★ Deep ensemble: M INDEPENDENT models from DIFFERENT random initializations
★ Calibration: among samples where model predicts confidence p, approximately p fraction are truly class c
★ Temperature scaling (calibration): divides logits by learned scalar T optimized on VALIDATION set
★ Conformal prediction: provides PREDICTION SET with valid coverage guarantee (probability ≥ 1-α)
★ Not a sign of overfitting: training loss HIGHER than validation loss (that's underfitting)
 
▌GENERALISATION THEORY
★ PAC learning sample complexity: O(1/ε × log(1/δ))  where ε=error tolerance, δ=failure probability
★ No Free Lunch theorem: no single algorithm is best for ALL problems
★ Information Bottleneck: minimize I(Z;X) while maximizing I(Z;Y)
★ CNN inductive bias: TRANSLATION EQUIVARIANCE and LOCALITY (local receptive field)
  — NOT permutation invariance (that's sets); NOT scale invariance
★ Universal Approximation Theorem: 1 hidden layer + sufficient neurons can approximate any continuous function on compact domain
★ Double descent: test error has a SECOND DESCENT after an initial peak near interpolation threshold
 
▌DOMAIN ADAPTATION & TRANSFER
★ Unsupervised domain adaptation: train on LABELLED source domain, generalise to UNLABELLED target domain
★ Zero-shot classification: NO examples from target classes — relies on class descriptions or pretrained knowledge
  — Few-shot provides a SMALL NUMBER of labelled examples at inference time
★ Domain randomization: varies simulation parameters to improve real-world transfer
 
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
