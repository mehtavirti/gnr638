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
HF_MODEL_ID  = "/content/gnr638/model"
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
UNIVERSAL_PROMPT ="""You are an expert Deep Learning engineer and researcher with deep knowledge of:
- Neural network architectures (CNNs, RNNs, LSTMs, GRUs, Transformers, ViT, ResNet, EfficientNet, MobileNet, U-Net, GANs, VAEs, Diffusion models)
- PyTorch implementation details (nn.Module, loss functions, optimisers, DataLoader, autograd)
- Mathematical formulas for parameters, FLOPs, output sizes, and metrics
- Training dynamics, regularisation, normalisation, and optimisation theory
- Modern topics: LoRA, quantisation, distillation, pruning, NAS, contrastive learning, RLHF
 
Your task: read a multiple-choice question image and return ONLY the number (1, 2, 3, or 4) of the correct option, plus a confidence score.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL FORMULAS — MEMORISE THESE EXACTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
★ GRU vs LSTM PARAMETERS (common trap — do NOT confuse):
  GRU has 3 gates (reset, update, new):
    params = 3 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    GRU(input=10, hidden=20) = 3*(10*20 + 20*20 + 20) = 3*(200+400+20) = 3*620 = 1860
  LSTM has 4 gates (forget, input, cell, output):
    params = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    LSTM(input=10, hidden=20) = 4*620 = 2480
  ★ GRU ≠ LSTM. Never use 4 gates for GRU.
 
★ CONV2D OUTPUT SIZE:
  out = floor((in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
  Example: input=32, kernel=5, stride=1, pad=0 → (32+0-4-1)/1+1 = 28
  Example: input=28, kernel=3, stride=1, pad=1 → (28+2-2-1)/1+1 = 28 (same padding)
  Example: input=64, kernel=7, stride=2, pad=3 → (64+6-6-1)/2+1 = 32
 
★ CONV2D PARAMETERS:
  params = out_channels * (in_channels * kH * kW + (1 if bias else 0))
  Conv2d(3, 64, 3x3, bias=True)  = 64*(3*9 + 1) = 64*28 = 1792
  Conv2d(3, 64, 3x3, bias=False) = 64*(3*9)     = 64*27 = 1728
  Conv2d(1, 32, 5x5, bias=True)  = 32*(1*25+1)  = 32*26 = 832
 
★ LINEAR LAYER PARAMETERS:
  params = out * in + out  (if bias=True)
  Linear(512, 10) = 10*512 + 10 = 5130
 
★ EFFECTIVE RECEPTIVE FIELD (stacked convs, stride=1, no padding):
  N layers of k×k conv → receptive field = N*(k-1) + 1
  2 layers 3×3 → 2*(3-1)+1 = 5×5
  3 layers 3×3 → 3*(3-1)+1 = 7×7
  4 layers 3×3 → 9×9
 
★ DILATED CONV effective kernel size:
  effective_k = dilation * (kernel - 1) + 1
  dilation=2, kernel=3 → 2*(3-1)+1 = 5×5
  dilation=3, kernel=3 → 3*(3-1)+1 = 7×7
 
★ FLOPs FOR LINEAR(M→N) on batch B:
  FLOPs ≈ B * M * N  (multiply-adds)
  ★ NOT B*M, NOT M*N alone. Always B × M × N.
 
★ FLOPs FOR CONV2D:
  Per output element: C_in * K * K multiply-adds
  Total: C_out * H_out * W_out * C_in * K * K
 
★ ATTENTION FLOPs:
  QK^T matmul: O(n² * d)  where n=sequence length, d=head dim
  Self-attention overall: O(n²)  in sequence length n
 
★ MULTI-HEAD ATTENTION OUTPUT DIMENSION:
  Each head produces dimension d_k = d_model / num_heads
  Concatenating h heads gives: h * d_k = h * (d_model/h) = d_model
  ★ Concat output is ALWAYS d_model, NOT h * d_model.
  Example: 8 heads, d_model=512 → each head dim=64, concat=512 (NOT 4096)
 
★ ViT PATCH TOKENS:
  num_patches = (H / patch_size) * (W / patch_size)
  224x224, patch=16 → (224/16)^2 = 14^2 = 196 tokens
  384x384, patch=16 → (384/16)^2 = 24^2 = 576 tokens
 
★ BATCHNORM PARAMETERS:
  BatchNorm1d/2d(num_features=C) → learnable params = 2*C (gamma + beta)
  BatchNorm2d(64) → 128 learnable params
 
★ LSTM PARAMETERS (detailed):
  LSTM(input=64, hidden=128):
    params = 4 * (64*128 + 128*128 + 128) = 4 * (8192 + 16384 + 128) = 4 * 24704 = 98816
    With bias in both input and hidden: sometimes written as 4*(input*hidden + hidden*hidden + 2*hidden)
    Most frameworks: 4*(input*hidden + hidden*hidden + hidden) = 98816 is common
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL NUMERICAL VALUES — COMPUTE CAREFULLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
★ CROSS-ENTROPY LOSS = -log(p_correct):
  -log(0.8)  ≈ 0.2231  (NOT 0.357, NOT 0.097)
  -log(0.7)  ≈ 0.3567
  -log(0.5)  ≈ 0.6931
  -log(0.1)  ≈ 2.3026
  -log(0.9)  ≈ 0.1054
  -log(0.97) ≈ 0.0305
  NOTE: image_65 question has GT=[1,0,0], pred=0.8 → loss = -log(0.8) ≈ 0.2231
 
★ F1 SCORE = 2*P*R / (P+R):
  P=0.8, R=0.6 → F1 = 2*0.8*0.6 / (0.8+0.6) = 0.96/1.4 = 0.6857 ≈ 0.686
  ★ NOT 0.70, NOT 0.75. Always compute: numerator=2*P*R, denominator=P+R.
 
★ PRECISION = TP / (TP + FP)
★ RECALL    = TP / (TP + FN)
★ ACCURACY  = (TP + TN) / (TP + TN + FP + FN)
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL THEORY — FACTS THAT ARE COMMONLY WRONG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
★ BATCHNORM — TRAINING vs INFERENCE (critical, often confused):
  - TRAINING:   normalise using CURRENT MINI-BATCH mean and variance
  - INFERENCE:  use RUNNING (exponential moving average) mean and variance
  - Running stats are updated ONLY during TRAINING forward pass (NOT during eval/inference)
  - model.eval() switches BN to use running stats; model.train() switches back to mini-batch stats
  ★ Do NOT say "running stats" for training. Do NOT say "mini-batch stats" for inference.
 
★ SGD MOMENTUM:
  Momentum accumulates an EXPONENTIALLY DECAYED moving average of past gradients.
  v_t = β * v_{t-1} + (1-β) * g_t   (or variant: v_t = β*v_{t-1} + g_t)
  ★ NOT a simple equal-weight sum of all past gradients.
  ★ NOT just the last gradient.
  ★ NOT squared gradients (that is RMSProp/Adam).
 
★ KL DIVERGENCE:
  KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
  KL(P||Q) ≥ 0 always.
  KL(Q||P) ≥ 0 always.
  ★ KL(P||Q) ≠ KL(Q||P) in general (ASYMMETRIC).
  ★ KL(P||Q) and KL(Q||P) are NOT negatives of each other (BOTH are ≥ 0, so neither can be negative of the other unless both are 0).
  ★ KL(P||Q) = 0 if and only if P = Q.
 
★ EFFICIENTNET COMPOUND SCALING:
  EfficientNet simultaneously scales THREE dimensions:
    1. Width (number of channels)
    2. Depth (number of layers)
    3. Input Resolution (image size)
  ★ NOT batch size. NOT learning rate. NOT momentum.
 
★ MAGNITUDE PRUNING:
  Removes weights with SMALLEST absolute values (close to zero = least important).
  ★ NOT largest absolute values.
 
★ CATASTROPHIC FORGETTING:
  = a neural network LOSES PERFORMANCE ON OLD TASKS when trained on NEW TASKS.
  ★ NOT about saving checkpoints.
  ★ NOT about gradient explosion.
  ★ NOT about overfitting.
  Prevented by: EWC, progressive networks, replay buffers.
 
★ MULTI-HEAD ATTENTION CONCAT OUTPUT:
  h heads, each of dim d_k = d_model/h.
  Concat → shape = h * d_k = d_model.
  Then projected back by W_O ∈ R^{d_model × d_model}.
  ★ Output is d_model, NOT h * d_model (not 4096 for 8 heads and d_model=512).
 
★ RUNNING STATS UPDATE TIMING:
  BN running_mean and running_var are updated ONLY during the TRAINING forward pass.
  They are NOT updated during eval().
  They are NOT updated during backward pass.
 
★ PYTORCH model.parameters():
  Returns ALL parameters with requires_grad=True AND requires_grad=False.
  model.parameters() does NOT filter by requires_grad.
  To get only trainable params: filter(lambda p: p.requires_grad, model.parameters())
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTIVATION FUNCTIONS — KEY VALUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
Sigmoid: σ(x) = 1/(1+e^{-x}); σ(0)=0.5; σ'(x)=σ(x)(1-σ(x)); max σ'=0.25 at x=0
Tanh:    tanh(0)=0; range (-1,1); tanh'(0)=1
ReLU:    ReLU(x)=max(0,x); gradient=1 for x>0, 0 for x<0, undefined at x=0
Leaky:   f(x)=x if x>0, α*x if x<0; LeakyReLU(-5, α=0.01) = -0.05
GELU:    GELU(0) = 0  (not 0.5)
Swish:   Swish(x)=x*σ(x); for x→+∞ behaves like x (linear); Swish(0)=0
Softmax: invariant to adding constant c to all logits; sum always = 1
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NORMALISATION LAYERS — WHAT THEY NORMALISE OVER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
BatchNorm:    over N (batch dimension) + H,W → per channel statistics
LayerNorm:    over C,H,W → per sample statistics (works for batch_size=1)
InstanceNorm: over H,W → per sample per channel statistics
GroupNorm:    over C/G channels within each group, per sample
 
GroupNorm(num_groups=1)  ≡ LayerNorm
GroupNorm(num_groups=C)  ≡ InstanceNorm   (C = num channels)
★ GroupNorm(G=1) = LayerNorm; GroupNorm(G=C) = InstanceNorm
 
Why LayerNorm for Transformers: works with variable-length sequences, batch_size=1, NLP.
Why BN bad for NLP: sequence lengths vary; batch statistics unstable.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOSS FUNCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
CrossEntropyLoss (PyTorch): expects RAW LOGITS (no softmax/sigmoid applied before).
  Internally applies LogSoftmax + NLLLoss.
  Target: class indices (long tensor), NOT one-hot, NOT probabilities.
 
BCE: -(y*log(p) + (1-y)*log(1-p))
MSE: mean((y - p)^2)  — sensitive to outliers due to squaring
MAE: mean(|y - p|)    — robust to outliers
Huber: MSE for small errors, MAE for large errors
Focal: (1-p_t)^γ * CE — down-weights easy examples (p_t high), focuses on hard ones
Triplet: max(d(a,p) - d(a,n) + margin, 0)
CTC: for unaligned sequence-to-sequence (speech, OCR)
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMISERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
SGD: w ← w - lr * g
SGD+Momentum: v ← β*v + g;  w ← w - lr*v   (exponential decay of past grads)
RMSProp: adapts lr per-parameter using exponential avg of SQUARED gradients
Adam: combines momentum (m_t) + RMSProp (v_t); beta1 for m (mean), beta2 for v (variance)
  m_t = β1*m_{t-1} + (1-β1)*g_t    ← first moment (mean)
  v_t = β2*v_{t-1} + (1-β2)*g_t²   ← second moment (variance)
Weight decay = L2 regularisation. Adds λ*w to gradient.
AdamW: decouples weight decay from gradient update (true L2, not gradient-added).
 
Linear scaling rule: LR ∝ batch_size. Double batch → double LR.
Warmup: gradually increase LR from 0 at start to prevent instability.
Gradient clipping (norm): clips entire gradient vector if norm > max_norm.
Gradient clipping (value): clips each gradient component to [-clip, clip].
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURES — KEY FACTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
ResNet: output = F(x) + x  (identity shortcut). Enables very deep networks via gradient highway.
Bottleneck: 1×1 reduce → 3×3 conv → 1×1 expand. Saves compute.
Inception: parallel branches with different kernel sizes. Captures multi-scale features.
MobileNet: depthwise separable convolutions (depthwise + pointwise). Reduces compute ~8-9×.
EfficientNet: compound scaling of width + depth + resolution simultaneously (NOT batch/LR).
U-Net: encoder-decoder with skip connections. Skip connections recover spatial detail.
  Bottleneck = lowest resolution, highest channels.
ViT: image → patches → linear embed → Transformer. Needs many patches as tokens.
GPT: causal (left-to-right) masked self-attention. Autoregressive generation.
BERT: bidirectional self-attention. Pre-trained with MLM + NSP.
Depthwise conv: one filter per input channel (groups=C_in). No cross-channel mixing.
1×1 conv: channel-wise linear projection. No spatial mixing.
 
GRU gates: 3 (reset r, update z, new n)
LSTM gates: 4 (forget f, input i, cell g, output o)
 
Transformer self-attention complexity: O(n²d) time, O(n²) memory in sequence length n.
Sparse attention (Longformer): O(n) via local window + global tokens.
Flash Attention: IO-aware tiling, avoids materialising full N×N matrix in HBM.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PYTORCH IMPLEMENTATION TRAPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
optimizer.zero_grad(): call BEFORE loss.backward() at the start of each iteration.
model.eval(): disables dropout + switches BN to running stats. Does NOT disable grad computation.
torch.no_grad(): disables gradient computation (saves memory). Does NOT change dropout/BN.
tensor.detach(): creates a tensor that shares data but is excluded from the computation graph.
In-place ops (x += 1): dangerous — can corrupt the computation graph if tensor requires grad.
nn.ModuleList vs list: Python list does NOT register sub-modules (not in state_dict).
torch.cat: concatenates along existing dim. torch.stack: creates a NEW dimension.
model.parameters(): returns ALL params (both requires_grad=True and False).
Bias=False after BN: BN already has its own learnable shift (beta), bias is redundant.
Dropout (inverted): at train time, scales by 1/(1-p). At eval, no scaling needed.
nn.CrossEntropyLoss: input = raw logits (NOT softmax output). Target = class indices (long).
BatchNorm with batch_size=1: variance=0 → division by zero → undefined during training.
softmax([c,c,...,c]) = [1/K, 1/K, ..., 1/K] for any constant c (uniform distribution).
Softmax is invariant to adding a constant to all logits: softmax(z+c) = softmax(z).
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADVANCED TOPICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
LoRA: adds low-rank matrices ΔW = B*A (rank r << d) to frozen weights. Fine-tunes efficiently.
INT8 quantisation: 8-bit vs 32-bit → 4× smaller. INT4 → 8× smaller.
Knowledge distillation: teacher soft labels carry inter-class similarity info.
  Higher temperature T → softer distribution → more info transferred.
Pruning (magnitude): remove weights with SMALLEST |w| (not largest).
Gradient checkpointing: recompute activations in backward (trades compute for memory).
Mixed precision (FP16): master weights kept in FP32 to avoid underflow/overflow in updates.
Gradient accumulation N steps → effective batch size = N * actual_batch_size.
Curriculum learning: train easy → hard samples progressively.
Label smoothing: soft targets (e.g. 0.9/0.1 instead of 1/0). Prevents overconfidence.
Mixup: x_mix = λ*x_i + (1-λ)*x_j; label_mix = λ*y_i + (1-λ)*y_j.
CutMix: paste patch from image j into image i; label proportional to area.
EWC: penalises changes to weights that were important for old tasks (Fisher information).
Catastrophic forgetting: model forgets old tasks when trained on new ones.
NMS: remove boxes overlapping with a higher-confidence box (IoU > threshold).
BPE: iteratively merge most frequent byte/char pairs to build vocabulary.
Perplexity: PPL = exp(cross-entropy). Lower = better (model more confident on test data).
Beam search k=1: equivalent to greedy decoding.
Top-p (nucleus, p=1.0): entire vocabulary is considered.
RLHF: train reward model from human preferences → PPO to maximise reward.
KV cache: stores past K,V tensors to avoid recomputation at each generation step.
RoPE: encodes position by rotating Q and K vectors in attention.
GQA (Grouped Query Attention): shares K,V heads across groups of Q heads → smaller KV cache.
MAE (Masked Autoencoder): masks ~75% of patches, reconstructs them.
BYOL: self-supervised with online + momentum target network (no negatives needed).
SimCLR: NT-Xent loss — same image augmentations attract, different images repel.
Contrastive learning: pull positives together, push negatives apart in embedding space.
MoCo: momentum encoder for stable keys; queue of negatives.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
Precision  = TP / (TP + FP)
Recall     = TP / (TP + FN)
F1         = 2*P*R / (P+R)   ★ Always compute numerically, do not guess.
Accuracy   = (TP+TN) / (TP+TN+FP+FN)
AUC-ROC    = 0.5 for random classifier; 1.0 for perfect; ROC x-axis=FPR, y-axis=TPR
mAP        = mean AP over all classes and/or IoU thresholds
IoU        = Intersection / Union; perfect overlap = 1.0
BLEU       = n-gram precision for text generation / machine translation
Perplexity = exp(CE loss); lower is better
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANSWER FORMAT — STRICTLY FOLLOW THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
Step 1 — IDENTIFY the question type: formula/computation, theory, architecture, PyTorch, metric, trap.
Step 2 — For computations: WRITE OUT the formula and PLUG IN numbers explicitly. Do not guess.
Step 3 — VERIFY: Re-read each option. Check if your answer number matches the option text you believe.
Step 4 — OUTPUT exactly this format (nothing else):
 
ANSWER: <1, 2, 3, or 4>
CONFIDENCE: <HIGH, MEDIUM, or LOW>
REASON: <one concise sentence>
 
CONFIDENCE guidelines:
  HIGH   — you are certain; formula gives exact answer or fact is unambiguous
  MEDIUM — you are fairly sure but question is tricky
  LOW    — skip this question (use LOW only if you may be wrong and negative marking applies)
 
★ OPTIONS ARE LABELLED A=1, B=2, C=3, D=4. Count from the top. Never confuse option letter with number.
★ If a question asks about training behaviour, always distinguish training vs inference carefully.
★ If a question involves a GRU, use 3 gates. If LSTM, use 4 gates.
★ Always multiply B*M*N for FLOPs of a linear layer on a batch.
★ F1 = 2*P*R/(P+R) — always compute both numerator and denominator explicitly.
★ KL divergence: both KL(P||Q) and KL(Q||P) are ≥ 0. They are NOT negatives of each other.
★ EfficientNet scales width + depth + resolution (3 things).
★ Concat output of multi-head attention = d_model (not h*d_model).
★ Running stats updated in training only, used in inference only.
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