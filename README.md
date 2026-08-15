# Deep Learning Visual MCQ Solver

An AI-powered system for solving image-based multiple-choice questions in deep learning and related mathematical topics using a vision-language model.

## Overview

The system takes images containing multiple-choice questions with four options and predicts the correct option automatically.

The pipeline combines:

- Image preprocessing for improved question readability
- Vision-language model based reasoning
- Domain-specific deep learning and mathematics knowledge
- Confidence-based answer filtering
- Robust answer extraction
- 4-bit model quantization for memory-efficient inference
- Automatic generation of a submission CSV

The model is instructed to carefully read all four options, solve the question, verify the alternatives, and provide a confidence estimate before selecting an answer.

## Pipeline

```text
Question Image
      │
      ▼
Image Preprocessing
      │
      ├── Auto-inversion
      ├── Deskewing
      ├── Whitespace cropping
      ├── Super-resolution
      ├── Contrast enhancement
      └── Sharpness enhancement
      │
      ▼
Qwen2.5-VL
      │
      ├── Read question and options
      ├── Apply relevant formulas/facts
      ├── Solve the question
      ├── Verify all options
      └── Estimate confidence
      │
      ▼
Confidence Gate
      │
      ├── LOW → Skip / output 5
      └── HIGH/MEDIUM
             │
             ▼
      Answer Extraction
             │
             ▼
      Option 1 / 2 / 3 / 4
             │
             ▼
       submission.csv
