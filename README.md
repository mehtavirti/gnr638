# GNR638 Assignment 2 — Pre-trained CNN Representation Transfer and Robustness Analysis

**Group Members**
- Virti Mehta (22b3949)
- Yajan Agarwal (22b1515)

---

## What this project does

We systematically study how three pre-trained CNN backbones — ResNet-50, DenseNet-121, and EfficientNet-B0 — behave on the Aerial Images Dataset (AID) with 30 land-use classes. The experiments cover linear probe transfer, fine-tuning strategies, few-shot learning, corruption robustness, and layer-wise feature probing.

---

## Project Structure
```
gnr638/
├── configs/                  # model and training configs
├── models/
│   └── load_models.py        # model loading, freezing, efficiency metrics
├── utils/
│   ├── dataset_loader.py     # AID dataset loader, train/val split, PCA subset
│   ├── metrics.py            # accuracy, corruption error, robustness metrics
│   └── corruption.py         # Gaussian noise, motion blur, brightness shift
├── evaluation/
│   └── evaluate.py           # main evaluation script (auto-downloads weights)
├── scenario_1.ipynb      # Linear Probe experiments
├── scenerio_2.ipynb      # Layer-Wise Feature Probing experiments
├── scenario_3.ipynb
├── scenario_4.ipynb
├── scenario_5.ipynb 
├── results/                  # will save plots and tables when run evaluate
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone -b assignment2 https://github.com/mehtavirti/gnr638.git
cd gnr638
```

### 2. Install dependencies

No virtual environment needed — just run:
```bash
pip install -r requirements.txt
```

This installs PyTorch, timm, scikit-learn, matplotlib, seaborn, pandas, and everything else needed.

> **Note:** The evaluation script also auto-installs `gdown` if it's missing, so you don't need to worry about that separately.

### 3. Dataset

The evaluation script works on any dataset in **ImageFolder format** — meaning a folder where each subfolder is a class name containing images.
```
your_dataset/
├── Airport/
│   ├── img1.jpg
│   └── ...
├── Beach/
│   └── ...
└── ... (30 classes)
```

Point `--data` to this folder when running evaluation.

---

## Running Evaluation

Model weights are stored on Google Drive and **downloaded automatically** the first time you run the script. You don't need to download anything manually.

### Evaluate all scenarios at once
```bash
python evaluation/evaluate.py --data /path/to/test_dataset --scenario all
```

### Evaluate a specific scenario
```bash
# Scenario 1 — Linear Probe Transfer
python evaluation/evaluate.py --data /path/to/test_dataset --scenario 1

# Scenario 2 — Fine-Tuning Strategies
python evaluation/evaluate.py --data /path/to/test_dataset --scenario 2

# Scenario 3 — Few-Shot Learning
python evaluation/evaluate.py --data /path/to/test_dataset --scenario 3

# Scenario 4 — Corruption Robustness
python evaluation/evaluate.py --data /path/to/test_dataset --scenario 4

# Scenario 5 — Layer-Wise Feature Probing
python evaluation/evaluate.py --data /path/to/test_dataset --scenario 5
```
### Drive links to all models and plots
scenario-1:https://drive.google.com/drive/folders/1pFgPqJ8jwDArULoG41lkDTNg0p4_aOdh?usp=drive_link
scenario-2:https://drive.google.com/drive/folders/1RVW2sZajagJGZx4Zy6cP8opYsuD-5a7U?usp=drive_link
scenario-3:https://drive.google.com/drive/folders/16KWZDPvEN550op1xZg5Cxjbs2asE08gR?usp=sharing
scenario-4:https://drive.google.com/drive/folders/1XT5v5FTT_Tw5Bufgxq_PvgwWfwuLFlTw?usp=drive_link
scenario-5:https://drive.google.com/drive/folders/1QqB70-CZc1gCkd8_KUek-qdl4fHvLKi4?usp=drive_link
