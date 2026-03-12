#!/usr/bin/env python3
# GNR638 Assignment 2 — Evaluation Script
# Usage:
#   python evaluation/evaluate.py --data /path/to/test_dataset --scenario 1
#   python evaluation/evaluate.py --data /path/to/test_dataset --scenario all

import os, sys, argparse, subprocess, importlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

def install_if_missing():
    for module, pip_name in [('gdown', 'gdown')]:
        try:
            importlib.import_module(module)
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name, '-q'])

install_if_missing()

import gc
import gdown
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

from models.load_models import (load_model, freeze_backbone, unfreeze_last_block,
                                 compute_efficiency_metrics, count_parameters)
from utils.dataset_loader import get_dataloaders
from utils.corruption import get_corruption_transform
from utils.metrics import corruption_error, relative_robustness

# =============================================================================
# CHECKPOINT IDs
# =============================================================================
CHECKPOINT_IDS = {
    'scenario1': {
        'resnet50':        '1KJUSNFUPJ1tYA8Y8EjhmPD_IaG_n836f',
        'densenet121':     '1JeViMCW6qNYAKAbPx6UohK5e85szxBRF',
        'efficientnet_b0': '1jKbcw5Cv9odjJNj0rEFuNirgtlJbpHkT',
    },
    'scenario2': {
        'resnet50': {
            'linear_probe':  '15SEEmWVw9oC4glp2F-CaPJ8hFuJ9bWaV',
            'last_block':    '12bqIoJo5rTFs1lNxQT3HfbQ8cE3E6rW4',
            'full_finetune': '1w6mFh9X9ZEbGC_FzgN62Qp035q674DsE',
            'selective20':   '1_iGLB62f0y7DiFacsDYXZVMWtWQHUL8L',
        },
        'densenet121': {
            'linear_probe':  '1zPc7qTUU_0B7SdAhVf6AINCKPg1dGGD6',
            'last_block':    '1freLTJIU3qUVEeQtLrcn5dPZxJmOdPH1',
            'full_finetune': '1wrOCHBWqHMaYDboptixiu0Q0FGaZQIRC',
            'selective20':   '1Jlx1NwJS_1epD1tm_mkziFAJTSjRZQen',
        },
        'efficientnet_b0': {
            'linear_probe':  '1qCFiqsgVZACFvqv6XJmNNwhK5UPiLatg',
            'last_block':    '1nPPaJaLpv94O27Fm2MM0A_NWTsTWONdF',
            'full_finetune': '1w_7d0c8bORGffeb_HbdLu8vbNYfJrrJj',
            'selective20':   '1n3sbugC6wEZfQvbx-iiDurM1alUxSOOP',
        },
    },
    # reuses resnet50_last_block, densenet121_last_block, efficientnet_b0_full_finetune
    'scenario4': {
        'resnet50':        '12bqIoJo5rTFs1lNxQT3HfbQ8cE3E6rW4',
        'densenet121':     '1freLTJIU3qUVEeQtLrcn5dPZxJmOdPH1',
        'efficientnet_b0': '1w_7d0c8bORGffeb_HbdLu8vbNYfJrrJj',
    },
    'scenario3': {
        'resnet50': {
            1.0:  '1Wu1FRuFsQMc0GFAYac2p3xN4q5NmRpr1',
            0.2:  '1TnMHZ7QbxzazB-jsPi9uSYNdkdjhEW3b',
            0.05: '1zi4j9OHXhcIf1SPz1VeFUkzWrb1Asc18',
        },
        'densenet121': {
            1.0:  '1BQ8eti6bgjnaylpxvmR9fFE4IwCeLyg0',
            0.2:  '1i-J0BtEgw2z0mDrGMZhmLr0GXxdn_yir',
            0.05: '162GE_zzZWrHD12V7TixuQ7_XkiVIDZ3F',
        },
        'efficientnet_b0': {
            1.0:  '1_DBbz2Cz-UGYg3pTib37VBJPt1QvUb1r',
            0.2:  '19vrrNr9mp7UMEhclwD1ZrRTfJcBrGAA_',
            0.05: '1Ew30xZuUukHwo159Oujv-av_BJs1I0iP',
        },
    },
    'scenario5': {
        'resnet50': {
            'early':  '1Vr1Wi_OWGeZaWYok9hwpdzo1k9qRUa4q',
            'middle': '1_4ECM3xskoyk4iz3jKymZeBSXUR8Yarl',
            'final':  '10ncbEmc0Euz81sUgyK4RGahhMoWduiPu',
        },
        'densenet121': {
            'early':  '1JtEe9Y_elbKcYww9N1sT4w3YL2xY4ltw',
            'middle': '1TisHCQgye-SzWTZnXWOeFfYaravY0PcY',
            'final':  '1yrFWc7cp-hLO34Xe2SbehKcRXXoYReWh',
        },
        'efficientnet_b0': {
            'early':  '168gWzQ1axY1pdQ0RuJzqfR_bQF3o4Lye',
            'middle': '1nyvlcB0ouLtBXwyk6n3x6y5dojWbB7dS',
            'final':  '1Xp4IauBArOr5PFBFVH9h_7Lg0kavep_K',
        },
    },
}

# =============================================================================
# CONSTANTS
# =============================================================================
NUM_CLASSES   = 30
MODEL_NAMES   = ['resnet50', 'densenet121', 'efficientnet_b0']
DEPTH_ORDER   = ['early', 'middle', 'final']
FRACTIONS     = [1.0, 0.2, 0.05]
S2_STRATEGIES = ['linear_probe', 'last_block', 'full_finetune', 'selective20']

BEST_STRATEGIES = {
    'resnet50':        'last_block',
    'densenet121':     'last_block',
    'efficientnet_b0': 'full_finetune',
}

S4_CORRUPTIONS = {
    'clean':         (None,          None),
    'gaussian_0.05': ('gaussian',    0.05),
    'gaussian_0.1':  ('gaussian',    0.1),
    'gaussian_0.2':  ('gaussian',    0.2),
    'motion_blur':   ('motion_blur', None),
    'brightness':    ('brightness',  0.2),
}

LAYER_SELECTION = {
    'resnet50':        {'early': 'layer1',               'middle': 'layer2',               'final': 'layer4'},
    'densenet121':     {'early': 'features.denseblock1',  'middle': 'features.denseblock2',  'final': 'features.denseblock4'},
    'efficientnet_b0': {'early': 'blocks.1',              'middle': 'blocks.3',              'final': 'blocks.6'},
}

COLORS = {
    'resnet50':        'steelblue',
    'densenet121':     'tomato',
    'efficientnet_b0': 'seagreen',
}

CKPT_DIR    = os.path.join(os.path.expanduser('~'), 'gnr638_checkpoints')
RESULTS_DIR = os.path.join(os.path.expanduser('~'), 'gnr638_results', 'evaluation')
for d in [CKPT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_file(file_id, save_path):
    if os.path.exists(save_path):
        print(f"  Already exists ({os.path.getsize(save_path)/1024**2:.1f} MB): {os.path.basename(save_path)}")
        return True
    if not file_id or 'PASTE_FILE_ID_HERE' in str(file_id):
        print(f"  File ID not set: {os.path.basename(save_path)}")
        return False
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"  Downloading: {os.path.basename(save_path)} ...")
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}&confirm=t",
                       save_path, quiet=False, fuzzy=True)
        print(f"    Done ({os.path.getsize(save_path)/1024**2:.1f} MB)")
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def download_scenario1_checkpoints():
    print("\n── Scenario 1 Checkpoints ───────────────────────────")
    paths = {}
    for name in MODEL_NAMES:
        path = os.path.join(CKPT_DIR, 'scenario1', f'{name}_best.pth')
        if download_file(CHECKPOINT_IDS['scenario1'][name], path):
            paths[name] = path
    print(f"  {len(paths)}/{len(MODEL_NAMES)} ready")
    return paths


def download_scenario2_checkpoints():
    print("\n── Scenario 2 Checkpoints ───────────────────────────")
    paths, success = {}, 0
    for name in MODEL_NAMES:
        paths[name] = {}
        for strategy in S2_STRATEGIES:
            path = os.path.join(CKPT_DIR, 'scenario2', f'{name}_{strategy}.pth')
            if download_file(CHECKPOINT_IDS['scenario2'][name][strategy], path):
                paths[name][strategy] = path
                success += 1
    print(f"  {success}/{len(MODEL_NAMES) * len(S2_STRATEGIES)} ready")
    return paths


def download_scenario3_checkpoints():
    print("\n── Scenario 3 Checkpoints ───────────────────────────")
    paths, success = {}, 0
    for name in MODEL_NAMES:
        paths[name] = {}
        for frac in FRACTIONS:
            path = os.path.join(CKPT_DIR, 'scenario3', f'{name}_fewshot_{frac}_best.pth')
            if download_file(CHECKPOINT_IDS['scenario3'][name][frac], path):
                paths[name][frac] = path
                success += 1
    print(f"  {success}/{len(MODEL_NAMES) * len(FRACTIONS)} ready")
    return paths


def download_scenario4_checkpoints():
    print("\n── Scenario 4 Checkpoints ───────────────────────────")
    paths = {}
    for name in MODEL_NAMES:
        path = os.path.join(CKPT_DIR, 'scenario4', f'{name}_{BEST_STRATEGIES[name]}.pth')
        if download_file(CHECKPOINT_IDS['scenario4'][name], path):
            paths[name] = path
    print(f"  {len(paths)}/{len(MODEL_NAMES)} ready")
    return paths


def download_scenario5_checkpoints():
    print("\n── Scenario 5 Checkpoints ───────────────────────────")
    paths, success = {}, 0
    for name in MODEL_NAMES:
        paths[name] = {}
        for depth in DEPTH_ORDER:
            path = os.path.join(CKPT_DIR, 'scenario5', f'{name}_{depth}_probe_best.pth')
            if download_file(CHECKPOINT_IDS['scenario5'][name][depth], path):
                paths[name][depth] = path
                success += 1
    print(f"  {success}/{len(MODEL_NAMES) * len(DEPTH_ORDER)} ready")
    return paths


# =============================================================================
# MODEL HELPERS
# =============================================================================

def _load_state(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    return {k: v for k, v in state.items()
            if not k.endswith(('total_ops', 'total_params'))}


def load_checkpoint(model_name, ckpt_path, device):
    model = load_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(_load_state(ckpt_path, device), strict=False)
    return model


def load_checkpoint_s2(model_name, ckpt_path, strategy, device):
    model = load_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    if strategy == 'linear_probe':
        model = freeze_backbone(model)
    elif strategy == 'last_block':
        model = freeze_backbone(model)
        model = unfreeze_last_block(model, model_name)
    elif strategy == 'full_finetune':
        for p in model.parameters():
            p.requires_grad = True
    elif strategy == 'selective20':
        model = _selective_unfreeze(model, target_ratio=0.20)
    model.load_state_dict(_load_state(ckpt_path, device), strict=False)
    return model


def load_checkpoint_s3(model_name, ckpt_path, strategy, device):
    model = load_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    if strategy == 'last_block':
        model = freeze_backbone(model)
        model = unfreeze_last_block(model, model_name)
    elif strategy == 'full_finetune':
        for p in model.parameters():
            p.requires_grad = True
    model.load_state_dict(_load_state(ckpt_path, device), strict=False)
    return model


def _selective_unfreeze(model, target_ratio=0.20):
    total = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for _, param in reversed(list(model.named_parameters())):
        param.requires_grad = True
        trainable += param.numel()
        if trainable / total >= target_ratio:
            break
    return model


def print_efficiency_info(model_name, device):
    model   = load_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    model   = freeze_backbone(model)
    metrics = compute_efficiency_metrics(model, input_size=(1, 3, 224, 224), device='cpu')
    total_p, trainable_p = count_parameters(model)
    print(f"    params: {total_p/1e6:.2f}M total, {trainable_p/1e6:.4f}M trainable")
    del model
    return metrics


# =============================================================================
# DATA + INFERENCE
# =============================================================================

def get_test_dataloader(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=torch.cuda.is_available())
    print(f"\n  {len(dataset)} samples | {len(dataset.classes)} classes | {data_dir}")
    return loader, dataset.classes


def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, preds     = torch.max(model(imgs), 1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / total, np.array(all_preds), np.array(all_labels)


def run_inference_with_transform(model, loader, device, corrupt_transform=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            if corrupt_transform is not None:
                imgs = corrupt_transform(imgs)
            imgs, labels = imgs.to(device), labels.to(device)
            _, preds     = torch.max(model(imgs), 1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
    return correct / total


def per_class_accuracy(all_labels, all_preds, classes):
    present_labels = sorted(set(int(l) for l in all_labels))
    cm             = confusion_matrix(all_labels, all_preds, labels=present_labels)
    row_sums       = cm.sum(axis=1)
    per_cls_acc    = np.where(row_sums > 0, cm.diagonal() / np.maximum(row_sums, 1), 0.0)
    present_names  = [classes[i] for i in present_labels]
    return pd.DataFrame({'Class': present_names,
                         'Accuracy': np.round(per_cls_acc, 4)
                         }).sort_values('Accuracy', ascending=True)


def save_confusion_matrix(all_labels, all_preds, classes, title, save_path):
    present_labels = sorted(set(int(l) for l in all_labels))
    present_names  = [classes[i] for i in present_labels]
    cm  = confusion_matrix(all_labels, all_preds, labels=present_labels)
    fig, ax = plt.subplots(figsize=(max(10, len(present_names) * 0.6),
                                    max(8,  len(present_names) * 0.5)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_names, yticklabels=present_names,
                ax=ax, annot_kws={'size': 7})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title(title,        fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Confusion matrix saved: {save_path}")


# =============================================================================
# SCENARIO 1
# =============================================================================

def evaluate_scenario1(data_dir, batch_size, device):
    print(f"\n{'='*60}\n  SCENARIO 1 — Linear Probe\n{'='*60}")

    ckpt_paths = download_scenario1_checkpoints()
    if not ckpt_paths:
        print("  No checkpoints. Aborting."); return None

    loader, classes = get_test_dataloader(data_dir, batch_size)
    summary_records = []

    for model_name in MODEL_NAMES:
        if model_name not in ckpt_paths:
            print(f"\n  Skipping {model_name} — checkpoint missing"); continue

        print(f"\n{'─'*55}\n  Model: {model_name}\n{'─'*55}")
        eff   = print_efficiency_info(model_name, device)
        model = load_checkpoint(model_name, ckpt_paths[model_name], device).to(device)

        acc, preds, labels = run_inference(model, loader, device)
        print(f"\n    Accuracy: {acc:.4f}  ({acc*100:.2f}%)")

        df_cls   = per_class_accuracy(labels, preds, classes)
        csv_path = os.path.join(RESULTS_DIR, f's1_{model_name}_per_class.csv')
        df_cls.to_csv(csv_path, index=False)
        print(f"\n    Bottom-5:\n{df_cls.head(5).to_string(index=False)}")

        save_confusion_matrix(labels, preds, classes,
            f'S1 — {model_name}  (Acc={acc:.4f})',
            os.path.join(RESULTS_DIR, f's1_{model_name}_confusion.png'))

        summary_records.append({
            'Model': model_name, 'Accuracy': round(acc, 4),
            'Params (M)': eff.get('params_M', ''),
            'MACs (G)':   eff.get('macs_G',   ''),
            'FLOPs (G)':  eff.get('flops_G',  ''),
        })
        del model; torch.cuda.empty_cache()

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 's1_summary.csv'), index=False)
    print(f"\n── Scenario 1 Summary ────────────────────────────────")
    print(df_summary.to_string(index=False))
    return df_summary


# =============================================================================
# SCENARIO 2
# =============================================================================

def evaluate_scenario2(data_dir, batch_size, device):
    print(f"\n{'='*60}\n  SCENARIO 2 — Fine-Tuning Strategies\n{'='*60}")

    ckpt_paths = download_scenario2_checkpoints()
    if not ckpt_paths:
        print("  No checkpoints. Aborting."); return None

    loader, classes = get_test_dataloader(data_dir, batch_size)
    summary_records = []

    for model_name in MODEL_NAMES:
        print(f"\n{'─'*55}\n  Model: {model_name}\n{'─'*55}")
        eff = print_efficiency_info(model_name, device)

        for strategy in S2_STRATEGIES:
            if strategy not in ckpt_paths.get(model_name, {}):
                print(f"\n  Skipping {strategy} — checkpoint missing"); continue

            print(f"\n    Strategy: {strategy}")
            model = load_checkpoint_s2(
                model_name, ckpt_paths[model_name][strategy], strategy, device).to(device)

            total_p, trainable_p = count_parameters(model)
            pct = trainable_p / total_p * 100 if total_p > 0 else 0.0
            print(f"      Trainable: {trainable_p/1e6:.4f}M  ({pct:.2f}%)")

            acc, preds, labels = run_inference(model, loader, device)
            print(f"      Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")

            df_cls   = per_class_accuracy(labels, preds, classes)
            csv_path = os.path.join(RESULTS_DIR, f's2_{model_name}_{strategy}_per_class.csv')
            df_cls.to_csv(csv_path, index=False)
            print(f"      Bottom-5:\n{df_cls.head(5).to_string(index=False)}")

            summary_records.append({
                'Model': model_name, 'Strategy': strategy,
                'Accuracy':      round(acc, 4),
                'Trainable (M)': round(trainable_p / 1e6, 4),
                'Pct Unfrozen':  round(pct, 2),
                'Params (M)':    eff.get('params_M', ''),
                'MACs (G)':      eff.get('macs_G',   ''),
                'FLOPs (G)':     eff.get('flops_G',  ''),
            })
            del model; torch.cuda.empty_cache()

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 's2_summary.csv'), index=False)
    print(f"\n── Scenario 2 Summary ────────────────────────────────")
    print(df_summary.to_string(index=False))

    if df_summary.empty:
        return df_summary

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in MODEL_NAMES:
        sub = df_summary[df_summary['Model'] == name].sort_values('Pct Unfrozen')
        if sub.empty: continue
        ax.plot(sub['Pct Unfrozen'], sub['Accuracy'],
                marker='o', label=name, color=COLORS[name], linewidth=2)
    ax.set_xlabel('% Parameters Unfrozen'); ax.set_ylabel('Test Accuracy')
    ax.set_title('S2: Accuracy vs Parameters Unfrozen')
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 's2_acc_vs_pct_unfrozen.png'),
                dpi=150, bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_summary, x='Strategy', y='Accuracy', hue='Model',
                palette=list(COLORS.values()), ax=ax)
    ax.set_title('S2: Accuracy Across Fine-Tuning Strategies')
    ax.set_ylabel('Test Accuracy'); ax.set_xlabel('Strategy'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 's2_strategy_bar.png'),
                dpi=150, bbox_inches='tight'); plt.close()

    df_summary['AccPerParam'] = df_summary.apply(
        lambda r: r['Accuracy'] / r['Pct Unfrozen'] if r['Pct Unfrozen'] > 0 else 0, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in MODEL_NAMES:
        sub = df_summary[df_summary['Model'] == name].sort_values('Pct Unfrozen')
        if sub.empty: continue
        ax.plot(sub['Pct Unfrozen'], sub['AccPerParam'],
                marker='o', label=name, color=COLORS[name], linewidth=2)
    ax.set_xlabel('% Parameters Unfrozen'); ax.set_ylabel('Accuracy / % Unfrozen')
    ax.set_title('S2: Parameter Efficiency')
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 's2_param_efficiency.png'),
                dpi=150, bbox_inches='tight'); plt.close()

    return df_summary


# =============================================================================
# SCENARIO 3
# =============================================================================

def evaluate_scenario3(data_dir, batch_size, device):
    print(f"\n{'='*60}\n  SCENARIO 3 — Few-Shot Learning\n{'='*60}")

    ckpt_paths = download_scenario3_checkpoints()
    if not ckpt_paths:
        print("  No checkpoints. Aborting."); return None

    loader, classes = get_test_dataloader(data_dir, batch_size)
    summary_records = []
    metrics_records = []

    for model_name in MODEL_NAMES:
        print(f"\n{'─'*55}\n  Model: {model_name}  |  strategy: {BEST_STRATEGIES[model_name]}\n{'─'*55}")
        eff          = print_efficiency_info(model_name, device)
        acc_per_frac = {}

        for frac in FRACTIONS:
            if frac not in ckpt_paths.get(model_name, {}):
                print(f"\n  Skipping fraction {frac} — checkpoint missing"); continue

            print(f"\n    Fraction: {frac}  ({int(frac*100)}%)")
            model = load_checkpoint_s3(
                model_name, ckpt_paths[model_name][frac],
                BEST_STRATEGIES[model_name], device).to(device)

            acc, preds, labels = run_inference(model, loader, device)
            print(f"      Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
            acc_per_frac[frac] = acc

            df_cls   = per_class_accuracy(labels, preds, classes)
            csv_path = os.path.join(RESULTS_DIR, f's3_{model_name}_frac{frac}_per_class.csv')
            df_cls.to_csv(csv_path, index=False)
            print(f"      Bottom-5:\n{df_cls.head(5).to_string(index=False)}")

            summary_records.append({
                'Model': model_name, 'Fraction': frac, 'Accuracy': round(acc, 4),
                'Params (M)': eff.get('params_M', ''),
                'MACs (G)':   eff.get('macs_G',   ''),
                'FLOPs (G)':  eff.get('flops_G',  ''),
            })
            del model; torch.cuda.empty_cache()

        if len(acc_per_frac) == 3:
            acc100 = acc_per_frac[1.0]; acc5 = acc_per_frac[0.05]
            drop   = (acc100 - acc5) / acc100 if acc100 > 0 else 0.0
            metrics_records.append({
                'model': model_name,
                'acc_100pct':    round(acc100, 4),
                'acc_20pct':     round(acc_per_frac[0.2], 4),
                'acc_5pct':      round(acc5, 4),
                'relative_drop': round(drop, 4),
            })

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 's3_summary.csv'), index=False)
    metrics_df = pd.DataFrame(metrics_records)
    if not metrics_df.empty:
        metrics_df.to_csv(os.path.join(RESULTS_DIR, 's3_metrics.csv'), index=False)

    print(f"\n── Scenario 3 Summary ────────────────────────────────")
    print(df_summary.to_string(index=False))
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))

    if not df_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name in MODEL_NAMES:
            sub = df_summary[df_summary['Model'] == name].sort_values('Fraction')
            if sub.empty: continue
            ax.plot(sub['Fraction'], sub['Accuracy'],
                    marker='o', label=name, color=COLORS[name], linewidth=2)
        ax.set_xlabel('Dataset Fraction'); ax.set_ylabel('Test Accuracy')
        ax.set_title('S3: Accuracy vs Data Fraction')
        ax.legend(); ax.grid(True, alpha=0.4); ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 's3_val_acc_vs_fraction.png'),
                    dpi=150, bbox_inches='tight'); plt.close()

        x, width = np.arange(len(MODEL_NAMES)), 0.25
        fig, ax  = plt.subplots(figsize=(9, 5))
        for i, frac in enumerate(FRACTIONS):
            vals = []
            for name in MODEL_NAMES:
                row = df_summary[(df_summary['Model'] == name) & (df_summary['Fraction'] == frac)]
                vals.append(float(row['Accuracy'].values[0]) if not row.empty else 0.0)
            ax.bar(x + (i - 1) * width, vals, width, label=f'{int(frac*100)}%')
        ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES, rotation=15)
        ax.set_ylabel('Test Accuracy'); ax.set_title('S3: Accuracy Across Data Fractions')
        ax.legend(); ax.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 's3_acc_comparison.png'),
                    dpi=150, bbox_inches='tight'); plt.close()

    if not metrics_df.empty:
        fig, ax    = plt.subplots(figsize=(7, 4))
        bar_colors = [COLORS[m] for m in metrics_df['model']]
        bars       = ax.bar(metrics_df['model'], metrics_df['relative_drop'], color=bar_colors)
        ax.set_ylabel('(Acc_100 − Acc_5) / Acc_100')
        ax.set_title('S3: Relative Drop (100% → 5% data)')
        ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.4)
        for bar, val in zip(bars, metrics_df['relative_drop']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 's3_relative_drop.png'),
                    dpi=150, bbox_inches='tight'); plt.close()

    return df_summary


# =============================================================================
# SCENARIO 4
# =============================================================================

def evaluate_scenario4(data_dir, batch_size, device):
    print(f"\n{'='*60}\n  SCENARIO 4 — Corruption Robustness\n{'='*60}")
    for name, strategy in BEST_STRATEGIES.items():
        print(f"    {name:<22}  {strategy}")

    ckpt_paths = download_scenario4_checkpoints()
    if not ckpt_paths:
        print("  No checkpoints. Aborting."); return None

    loader, classes = get_test_dataloader(data_dir, batch_size)
    records = []

    for model_name in MODEL_NAMES:
        if model_name not in ckpt_paths:
            print(f"\n  Skipping {model_name} — checkpoint missing"); continue

        strategy = BEST_STRATEGIES[model_name]
        print(f"\n{'─'*55}\n  Model: {model_name}  ({strategy})\n{'─'*55}")

        model     = load_checkpoint_s2(model_name, ckpt_paths[model_name],
                                       strategy, device).to(device)
        clean_acc = None

        for cname, (ctype, severity) in S4_CORRUPTIONS.items():
            corrupt_tf = None if cname == 'clean' else get_corruption_transform(ctype, severity)
            acc        = run_inference_with_transform(model, loader, device, corrupt_tf)

            if cname == 'clean':
                clean_acc = acc

            ce  = corruption_error(acc)
            rr  = relative_robustness(acc, clean_acc) if clean_acc is not None else 1.0
            tag = 'baseline' if cname == 'clean' else f'CE={ce:.4f}'
            print(f"    {cname:<20}  Acc={acc:.4f}  {tag}")

            records.append({
                'Model':               model_name,
                'Corruption':          cname,
                'Accuracy':            round(acc, 4),
                'Corruption Error':    round(ce, 4),
                'Relative Robustness': round(rr, 4),
            })

        del model; torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS_DIR, 's4_summary.csv'), index=False)
    print(f"\n── Scenario 4 Summary ────────────────────────────────")
    print(df.to_string(index=False))

    if df.empty:
        return df

    df_corrupted = df[df['Corruption'] != 'clean']

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df_corrupted, x='Corruption', y='Accuracy',
                hue='Model', palette=list(COLORS.values()), ax=ax)
    ax.set_title('S4: Accuracy under Corruptions')
    ax.set_xlabel('Corruption'); ax.set_ylabel('Test Accuracy')
    plt.xticks(rotation=30, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 's4_accuracy_corruptions.png'),
                dpi=150, bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df_corrupted, x='Corruption', y='Relative Robustness',
                hue='Model', palette=list(COLORS.values()), ax=ax)
    ax.set_title('S4: Relative Robustness')
    ax.set_xlabel('Corruption'); ax.set_ylabel('Acc_corrupted / Acc_clean')
    plt.xticks(rotation=30, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 's4_relative_robustness.png'),
                dpi=150, bbox_inches='tight'); plt.close()

    pivot = df_corrupted.pivot(index='Model', columns='Corruption', values='Corruption Error')
    fig, ax = plt.subplots(figsize=(max(9, len(pivot.columns) * 1.4), 4))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Reds', linewidths=0.4, ax=ax)
    ax.set_title('S4: Corruption Error  (1 − Acc_corrupted)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 's4_corruption_error_heatmap.png'),
                dpi=150, bbox_inches='tight'); plt.close()

    return df


# =============================================================================
# SCENARIO 5
# =============================================================================

def extract_layer_features(backbone, loader, layer_name, device):
    named_mods = dict(backbone.named_modules())
    module     = named_mods.get(layer_name)
    if module is None:
        for n, m in named_mods.items():
            if n.startswith(layer_name):
                module = m; break
    if module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    captured = []
    def hook_fn(m, inp, out):
        feat = out.detach().cpu()
        if feat.dim() == 4:   feat = feat.mean(dim=[2, 3])
        elif feat.dim() == 3: feat = feat.mean(dim=1)
        captured.append(feat)

    hook = module.register_forward_hook(hook_fn)
    feats_list, labels_list = [], []
    backbone.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            captured.clear()
            _ = backbone(imgs.to(device))
            feats_list.append(captured[0])
            labels_list.extend(labels.numpy())
    hook.remove()
    feats = torch.cat(feats_list, dim=0)
    print(f"      Feature shape: {feats.shape}")
    return feats, np.array(labels_list)


def evaluate_scenario5(data_dir, batch_size, device):
    print(f"\n{'='*60}\n  SCENARIO 5 — Layer-Wise Probing\n{'='*60}")

    ckpt_paths = download_scenario5_checkpoints()
    if not ckpt_paths:
        print("  No checkpoints. Aborting."); return None

    loader, classes = get_test_dataloader(data_dir, batch_size)
    summary_records = []

    for model_name in MODEL_NAMES:
        print(f"\n{'─'*55}\n  Model: {model_name}\n{'─'*55}")
        eff      = print_efficiency_info(model_name, device)
        backbone = load_model(model_name, num_classes=NUM_CLASSES, pretrained=True)
        backbone = freeze_backbone(backbone).to(device)
        backbone.eval()

        for depth in DEPTH_ORDER:
            if depth not in ckpt_paths.get(model_name, {}):
                print(f"\n  Skipping {depth} — checkpoint missing"); continue

            layer_name = LAYER_SELECTION[model_name][depth]
            ckpt_path  = ckpt_paths[model_name][depth]
            print(f"\n    Layer: {depth} ({layer_name})")

            feats, labels = extract_layer_features(backbone, loader, layer_name, device)

            in_dim = feats.shape[1]
            clf    = nn.Linear(in_dim, NUM_CLASSES).to(device)
            clf.load_state_dict(_load_state(ckpt_path, device))
            clf.eval()

            feat_loader = DataLoader(
                TensorDataset(feats, torch.tensor(labels, dtype=torch.long)),
                batch_size=batch_size, shuffle=False,
                pin_memory=torch.cuda.is_available())

            correct, total, all_preds = 0, 0, []
            with torch.no_grad():
                for xb, yb in feat_loader:
                    xb, yb  = xb.to(device), yb.to(device)
                    _, pred = torch.max(clf(xb), 1)
                    correct += (pred == yb).sum().item()
                    total   += yb.size(0)
                    all_preds.extend(pred.cpu().numpy())

            acc = correct / total
            print(f"      Accuracy: {acc:.4f}  ({acc*100:.2f}%)")

            df_cls   = per_class_accuracy(labels, all_preds, classes)
            csv_path = os.path.join(RESULTS_DIR, f's5_{model_name}_{depth}_per_class.csv')
            df_cls.to_csv(csv_path, index=False)
            print(f"      Per-class CSV: {csv_path}")

            summary_records.append({
                'Model': model_name, 'Layer': depth, 'Layer Name': layer_name,
                'Accuracy': round(acc, 4), 'Feature Dim': in_dim,
                'Params (M)': eff.get('params_M', ''),
                'MACs (G)':   eff.get('macs_G',   ''),
                'FLOPs (G)':  eff.get('flops_G',  ''),
            })
            del clf; torch.cuda.empty_cache()

        del backbone; torch.cuda.empty_cache()

    df_summary = pd.DataFrame(summary_records)
    csv_path   = os.path.join(RESULTS_DIR, 's5_summary.csv')
    df_summary.to_csv(csv_path, index=False)

    if not df_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for name in MODEL_NAMES:
            sub = df_summary[df_summary['Model'] == name]
            if sub.empty: continue
            vals, depths_found = [], []
            for d in DEPTH_ORDER:
                row = sub[sub['Layer'] == d]
                if not row.empty:
                    vals.append(row['Accuracy'].values[0]); depths_found.append(d)
            ax.plot(depths_found, vals, marker='o', label=name,
                    color=COLORS[name], linewidth=2)
        ax.set_title('S5: Accuracy vs Layer Depth', fontsize=13)
        ax.set_xlabel('Layer Depth'); ax.set_ylabel('Accuracy')
        ax.legend(); ax.grid(True, alpha=0.4); ax.set_ylim(0, 1)
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, 's5_accuracy_vs_depth.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"\n  Plot saved: {plot_path}")

    print(f"\n── Scenario 5 Summary ────────────────────────────────")
    print(df_summary.to_string(index=False))
    print(f"  Summary: {csv_path}")
    return df_summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GNR638 A2 — Evaluation',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data',        required=True,
                        help='Path to test dataset (ImageFolder format)')
    parser.add_argument('--scenario',    default='all',
                        choices=['1', '2', '3', '4', '5', 'all'])
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--ckpt_dir',    default=None)
    parser.add_argument('--results_dir', default=None)
    args = parser.parse_args()

    global CKPT_DIR, RESULTS_DIR
    if args.ckpt_dir:    CKPT_DIR    = os.path.abspath(args.ckpt_dir)
    if args.results_dir: RESULTS_DIR = os.path.abspath(args.results_dir)
    for d in [CKPT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}\n  GNR638 A2 — Evaluation\n{'='*60}")
    print(f"  Dataset    : {args.data}")
    print(f"  Scenario   : {args.scenario}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Device     : {device}")
    print(f"  Checkpoints: {CKPT_DIR}")
    print(f"  Results    : {RESULTS_DIR}")
    print(f"{'='*60}")

    if not os.path.isdir(args.data):
        print(f"\n  Dataset path not found: {args.data}"); sys.exit(1)

    if args.scenario in ('1', 'all'): evaluate_scenario1(args.data, args.batch_size, device)
    if args.scenario in ('2', 'all'): evaluate_scenario2(args.data, args.batch_size, device)
    if args.scenario in ('3', 'all'): evaluate_scenario3(args.data, args.batch_size, device)
    if args.scenario in ('4', 'all'): evaluate_scenario4(args.data, args.batch_size, device)
    if args.scenario in ('5', 'all'): evaluate_scenario5(args.data, args.batch_size, device)

    print(f"\n── Output files ──────────────────────────────────────")
    for f in sorted(os.listdir(RESULTS_DIR)):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        unit, val = ('MB', size/1024**2) if size > 1e6 else ('KB', size/1024)
        print(f"  {f:<55} {val:>6.1f} {unit}")

    print(f"\n{'='*60}\n  Done. Results  {RESULTS_DIR}\n{'='*60}\n")


if __name__ == '__main__':
    main()