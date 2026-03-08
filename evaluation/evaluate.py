#!/usr/bin/env python3
# =============================================================================
# GNR638 Assignment 2 — Evaluation Script
# Grader usage:
#   git clone -b assignment2 https://github.com/mehtavirti/gnr638.git
#   cd gnr638
#   pip install -r requirements.txt
#   python evaluation/evaluate.py --data /path/to/test_dataset --scenario 1
#   python evaluation/evaluate.py --data /path/to/test_dataset --scenario 5
#   python evaluation/evaluate.py --data /path/to/test_dataset --scenario all
# =============================================================================

import os, sys, argparse, subprocess, importlib

# ── Add repo root to path so shared utils are importable ──────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ── Auto-install only what requirements.txt doesn't cover ─────────────────────
def install_if_missing():
    for module, pip_name in [('gdown', 'gdown')]:
        try:
            importlib.import_module(module)
        except ImportError:
            print(f"  Installing {pip_name}...")
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', pip_name, '-q'])

install_if_missing()

import gdown
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

# ── Import shared utilities from repo (available after git clone) ──────────────
from models.load_models import (load_model, freeze_backbone,
                                 compute_efficiency_metrics, count_parameters)
from utils.dataset_loader import get_dataloaders

CHECKPOINT_IDS = {

    # ── Scenario 1: full model checkpoints (trained linear probe) ─────────────
    'scenario1': {
        'resnet50':        '1KJUSNFUPJ1tYA8Y8EjhmPD_IaG_n836f',
        'densenet121':     '1JeViMCW6qNYAKAbPx6UohK5e85szxBRF',
        'efficientnet_b0': '1jKbcw5Cv9odjJNj0rEFuNirgtlJbpHkT',
       
    },

    # ── Scenario 5: nn.Linear layer classifiers (one per layer per model) ─────
    'scenario5': {
        'resnet50': {
            'early':  '1Vr1Wi_OWGeZaWYok9hwpdzo1k9qRUa4q',
            'middle': '1_4ECM3xskoyk4iz3jKymZeBSXUR8Yarl',
            'final':  '10ncbEmc0Euz81sUgyK4RGahhMoWduiPu',
        },
        'densenet121': {
            'early':  'PASTE_FILE_ID_HERE',
            'middle': 'PASTE_FILE_ID_HERE',
            'final':  'PASTE_FILE_ID_HERE',
        },
        'efficientnet_b0': {
            'early':  'PASTE_FILE_ID_HERE',
            'middle': 'PASTE_FILE_ID_HERE',
            'final':  'PASTE_FILE_ID_HERE',
        },
    },
}

# =============================================================================
# CONSTANTS
# =============================================================================
NUM_CLASSES = 30
MODEL_NAMES = ['resnet50', 'densenet121', 'efficientnet_b0']
DEPTH_ORDER = ['early', 'middle', 'final']
COLORS      = {'resnet50':        'steelblue',
               'densenet121':     'tomato',
               'efficientnet_b0': 'seagreen'}

LAYER_SELECTION = {
    'resnet50': {
        'early':  'layer1',
        'middle': 'layer2',
        'final':  'layer4',
    },
    'densenet121': {
        'early':  'features.denseblock1',
        'middle': 'features.denseblock2',
        'final':  'features.denseblock4',
    },
    'efficientnet_b0': {
        'early':  'blocks.1',
        'middle': 'blocks.3',
        'final':  'blocks.6',
    },
}

# ── Save checkpoints and results to user's home directory, NOT inside the repo ─
CKPT_DIR    = os.path.join(os.path.expanduser('~'), 'gnr638_checkpoints')
RESULTS_DIR = os.path.join(os.path.expanduser('~'), 'gnr638_results', 'evaluation')

for d in [CKPT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# DOWNLOAD UTILITIES — only .pth files, everything else from repo
# =============================================================================

def download_file(file_id, save_path):
    """Download one .pth from Google Drive. Skip if already exists."""
    if os.path.exists(save_path):
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f" Already exists ({size_mb:.1f} MB): "
              f"{os.path.basename(save_path)}")
        return True

    if not file_id or 'PASTE_FILE_ID_HERE' in file_id:
        print(f" File ID not set: {os.path.basename(save_path)}")
        print(f" Fill CHECKPOINT_IDS in evaluate.py with real Drive IDs")
        return False

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
    print(f" Downloading: {os.path.basename(save_path)} ...")
    try:
        gdown.download(url, save_path, quiet=False, fuzzy=True)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"   Done ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"   Download failed: {e}")
        return False


def download_scenario1_checkpoints():
    print("\n── Downloading Scenario 1 Checkpoints ───────────────")
    paths = {}
    for name in MODEL_NAMES:
        fid  = CHECKPOINT_IDS['scenario1'][name]
        path = os.path.join(CKPT_DIR, 'scenario1', f'{name}_best.pth')
        if download_file(fid, path):
            paths[name] = path
    print(f"  Ready: {len(paths)}/{len(MODEL_NAMES)} models")
    return paths


def download_scenario5_checkpoints():
    print("\n── Downloading Scenario 5 Checkpoints ───────────────")
    paths   = {}
    total   = len(MODEL_NAMES) * len(DEPTH_ORDER)
    success = 0
    for name in MODEL_NAMES:
        paths[name] = {}
        for depth in DEPTH_ORDER:
            fid  = CHECKPOINT_IDS['scenario5'][name][depth]
            path = os.path.join(CKPT_DIR, 'scenario5',
                                f'{name}_{depth}_probe_best.pth')
            if download_file(fid, path):
                paths[name][depth] = path
                success += 1
    print(f"  Ready: {success}/{total} layer classifiers")
    return paths


# =============================================================================
# MODEL UTILITIES — uses shared models/load_models.py from repo
# =============================================================================

def load_checkpoint(model_name, ckpt_path, device):
    """Load model using shared load_model(), strip thop keys."""
    model = load_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    clean = {k: v for k, v in state.items()
             if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(clean, strict=False)
    return model


def print_efficiency_info(model_name, device):
    """Print params/MACs/FLOPs using shared compute_efficiency_metrics."""
    print(f"\n    [Efficiency — {model_name}]")
    model  = load_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    model  = freeze_backbone(model)
    metrics = compute_efficiency_metrics(
        model, input_size=(1, 3, 224, 224), device='cpu')
    total_p, trainable_p = count_parameters(model)
    print(f"    Total params     : {total_p/1e6:.2f} M")
    print(f"    Trainable params : {trainable_p/1e6:.4f} M")
    del model
    return metrics


# =============================================================================
# DATA + INFERENCE UTILITIES
# =============================================================================

def get_test_dataloader(data_dir, batch_size=64):
    """Clean test transform — no augmentation."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=2, pin_memory=True)
    print(f"\n  Dataset : {len(dataset)} samples | "
          f"{len(dataset.classes)} classes")
    print(f"  Path    : {data_dir}")
    return loader, dataset.classes


def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out          = model(imgs)
            _, preds     = torch.max(out, 1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / total, np.array(all_preds), np.array(all_labels)


def per_class_accuracy(all_labels, all_preds, classes):
    cm          = confusion_matrix(all_labels, all_preds)
    per_cls_acc = cm.diagonal() / cm.sum(axis=1)
    return pd.DataFrame({
        'Class':    classes,
        'Accuracy': np.round(per_cls_acc, 4)
    }).sort_values('Accuracy', ascending=True)


def save_confusion_matrix(all_labels, all_preds, classes, title, save_path):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, annot_kws={'size': 7})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title(title,        fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Confusion matrix → {save_path}")


# =============================================================================
# SCENARIO 1 EVALUATION
# =============================================================================

def evaluate_scenario1(data_dir, batch_size, device):
    print(f"\n{'='*60}")
    print(f"  SCENARIO 1 — Linear Probe Transfer Evaluation")
    print(f"{'='*60}")

    ckpt_paths = download_scenario1_checkpoints()
    if not ckpt_paths:
        print(" No checkpoints available. Aborting Scenario 1.")
        return None

    loader, classes = get_test_dataloader(data_dir, batch_size)
    summary_records = []

    for model_name in MODEL_NAMES:
        if model_name not in ckpt_paths:
            print(f"\n Skipping {model_name} — checkpoint missing")
            continue

        print(f"\n{'─'*55}")
        print(f"  Model: {model_name}")
        print(f"{'─'*55}")

        # Efficiency (uses shared util)
        eff = print_efficiency_info(model_name, device)

        # Load trained model
        model = load_checkpoint(model_name, ckpt_paths[model_name], device)
        model = model.to(device)

        # Inference
        acc, preds, labels = run_inference(model, loader, device)
        print(f"\n    Overall Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

        # Per-class accuracy
        df_cls   = per_class_accuracy(labels, preds, classes)
        csv_path = os.path.join(RESULTS_DIR, f's1_{model_name}_per_class.csv')
        df_cls.to_csv(csv_path, index=False)
        print(f"\n    Bottom 5 failure classes:")
        print(df_cls.head(5).to_string(index=False))
        print(f"    Per-class CSV → {csv_path}")

        # Confusion matrix
        save_confusion_matrix(
            labels, preds, classes,
            f'Scenario 1 — {model_name}  (Acc: {acc:.4f})',
            os.path.join(RESULTS_DIR, f's1_{model_name}_confusion.png'))

        summary_records.append({
            'Model':      model_name,
            'Accuracy':   round(acc, 4),
            'Params (M)': eff.get('params_M', ''),
            'MACs (G)':   eff.get('macs_G',   ''),
            'FLOPs (G)':  eff.get('flops_G',  ''),
        })

        del model
        torch.cuda.empty_cache()

    # Summary
    df_summary = pd.DataFrame(summary_records)
    csv_path   = os.path.join(RESULTS_DIR, 's1_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"\n── Scenario 1 Summary ────────────────────────────────")
    print(df_summary.to_string(index=False))
    print(f"Summary → {csv_path}")
    return df_summary


# =============================================================================
# SCENARIO 5 EVALUATION
# =============================================================================

def extract_layer_features(backbone, loader, layer_name, device):
    """Extract GAP features from named intermediate layer via hook."""
    named_mods = dict(backbone.named_modules())
    module     = named_mods.get(layer_name)
    if module is None:
        for n, m in named_mods.items():
            if n.startswith(layer_name):
                module = m
                break
    if module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    captured = []
    def hook_fn(m, inp, out):
        feat = out.detach().cpu()
        if feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])
        elif feat.dim() == 3:
            feat = feat.mean(dim=1)
        captured.append(feat)

    hook        = module.register_forward_hook(hook_fn)
    feats_list  = []
    labels_list = []

    backbone.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            captured.clear()
            _  = backbone(imgs.to(device))
            feats_list.append(captured[0])
            labels_list.extend(labels.numpy())

    hook.remove()
    feats = torch.cat(feats_list, dim=0)
    print(f"      Feature shape: {feats.shape}")
    return feats, np.array(labels_list)


def evaluate_scenario5(data_dir, batch_size, device):
    print(f"\n{'='*60}")
    print(f"  SCENARIO 5 — Layer-Wise Feature Probing Evaluation")
    print(f"{'='*60}")

    ckpt_paths = download_scenario5_checkpoints()
    if not ckpt_paths:
        print(" No checkpoints available. Aborting Scenario 5.")
        return None

    loader, classes = get_test_dataloader(data_dir, batch_size)
    summary_records = []

    for model_name in MODEL_NAMES:
        print(f"\n{'─'*55}")
        print(f"  Model: {model_name}")
        print(f"{'─'*55}")

        # Efficiency
        eff = print_efficiency_info(model_name, device)

        # Frozen pretrained backbone — same setup as training
        backbone = load_model(model_name, num_classes=NUM_CLASSES, pretrained=True)
        backbone = freeze_backbone(backbone)
        backbone = backbone.to(device)
        backbone.eval()

        for depth in DEPTH_ORDER:
            if depth not in ckpt_paths.get(model_name, {}):
                print(f"\n  Skipping {depth} — checkpoint missing")
                continue

            layer_name = LAYER_SELECTION[model_name][depth]
            ckpt_path  = ckpt_paths[model_name][depth]

            print(f"\n    ── Layer: {depth} ({layer_name})")

            # Extract features using frozen backbone
            feats, labels = extract_layer_features(
                backbone, loader, layer_name, device)

            # Load saved nn.Linear classifier
            in_dim = feats.shape[1]
            clf    = nn.Linear(in_dim, NUM_CLASSES).to(device)
            ckpt   = torch.load(ckpt_path, map_location=device)
            state  = ckpt.get('state_dict', ckpt)
            clf.load_state_dict(state)
            clf.eval()

            # Evaluate
            feat_loader = DataLoader(
                TensorDataset(feats, torch.tensor(labels)),
                batch_size=batch_size, shuffle=False)

            correct, total = 0, 0
            all_preds = []
            with torch.no_grad():
                for xb, yb in feat_loader:
                    xb, yb  = xb.to(device), yb.to(device)
                    out     = clf(xb)
                    _, pred = torch.max(out, 1)
                    correct += (pred == yb).sum().item()
                    total   += yb.size(0)
                    all_preds.extend(pred.cpu().numpy())

            acc = correct / total
            print(f"      Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

            # Per-class CSV
            df_cls   = per_class_accuracy(labels, all_preds, classes)
            csv_path = os.path.join(RESULTS_DIR,
                                    f's5_{model_name}_{depth}_per_class.csv')
            df_cls.to_csv(csv_path, index=False)
            print(f"      ✓ Per-class CSV → {csv_path}")

            summary_records.append({
                'Model':       model_name,
                'Layer':       depth,
                'Layer Name':  layer_name,
                'Accuracy':    round(acc, 4),
                'Feature Dim': in_dim,
                'Params (M)':  eff.get('params_M', ''),
                'MACs (G)':    eff.get('macs_G',   ''),
                'FLOPs (G)':   eff.get('flops_G',  ''),
            })

            del clf
            torch.cuda.empty_cache()

        del backbone
        torch.cuda.empty_cache()

    # Summary table
    df_summary = pd.DataFrame(summary_records)
    csv_path   = os.path.join(RESULTS_DIR, 's5_summary.csv')
    df_summary.to_csv(csv_path, index=False)

    # Accuracy vs depth plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODEL_NAMES:
        sub = df_summary[df_summary['Model'] == name]
        if sub.empty:
            continue
        vals, depths_found = [], []
        for d in DEPTH_ORDER:
            row = sub[sub['Layer'] == d]
            if not row.empty:
                vals.append(row['Accuracy'].values[0])
                depths_found.append(d)
        ax.plot(depths_found, vals, marker='o',
                label=name, color=COLORS[name], linewidth=2)

    ax.set_title('Scenario 5 — Accuracy vs Layer Depth', fontsize=13)
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.4); ax.set_ylim(0, 1)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 's5_accuracy_vs_depth.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n Accuracy vs depth plot → {plot_path}")

    print(f"\n── Scenario 5 Summary ────────────────────────────────")
    print(df_summary.to_string(index=False))
    print(f" Summary {csv_path}")
    return df_summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GNR638 A2 — Evaluation Script',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python evaluation/evaluate.py --data /path/to/test_dataset
  python evaluation/evaluate.py --data /path/to/test_dataset --scenario 1
  python evaluation/evaluate.py --data /path/to/test_dataset --scenario 5
        """)

    parser.add_argument('--data',       required=True,
                        help='Path to test dataset (ImageFolder format)')
    parser.add_argument('--scenario',   default='all',
                        choices=['1', '5', 'all'],
                        help='Scenario to evaluate: 1, 5, or all (default: all)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference (default: 64)')
    parser.add_argument('--ckpt_dir',   default=None,
                        help='Override checkpoint directory '
                             '(default: ~/gnr638_checkpoints)')
    parser.add_argument('--results_dir', default=None,
                        help='Override results directory '
                             '(default: ~/gnr638_results/evaluation)')
    args = parser.parse_args()

    # Allow CLI overrides for paths
    global CKPT_DIR, RESULTS_DIR
    if args.ckpt_dir:
        CKPT_DIR = os.path.abspath(args.ckpt_dir)
    if args.results_dir:
        RESULTS_DIR = os.path.abspath(args.results_dir)
    for d in [CKPT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Evaluation")
    print(f"{'='*60}")
    print(f"  Dataset      : {args.data}")
    print(f"  Scenario     : {args.scenario}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Device       : {device}")
    print(f"  Checkpoints → : {CKPT_DIR}")
    print(f"  Results    → : {RESULTS_DIR}")
    print(f"{'='*60}")

    if not os.path.isdir(args.data):
        print(f"\n  Dataset path not found: {args.data}")
        sys.exit(1)

    if args.scenario in ('1', 'all'):
        evaluate_scenario1(args.data, args.batch_size, device)

    if args.scenario in ('5', 'all'):
        evaluate_scenario5(args.data, args.batch_size, device)

    # Print all saved output files
    print(f"\n── All output files ──────────────────────────────────")
    for f in sorted(os.listdir(RESULTS_DIR)):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        unit, val = ('MB', size/1024**2) if size > 1e6 else ('KB', size/1024)
        print(f"  {f:<55} {val:>6.1f} {unit}")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Checkpoints : {CKPT_DIR}")
    print(f"  Results     : {RESULTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()