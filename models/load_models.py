import torch
import torch.nn as nn
import timm
from thop import profile
from fvcore.nn import FlopCountAnalysis


def load_model(model_name, num_classes=30, pretrained=True):
    """
    Load pretrained model from timm and replace classifier head.
    Works for: resnet50, densenet121, efficientnet_b0
    """
    model = timm.create_model(model_name, pretrained=pretrained)

    # Replace classifier head based on architecture
    if hasattr(model, 'fc'):                              # ResNet50
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif hasattr(model, 'classifier'):                    # DenseNet, EfficientNet
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
        else:
            in_features = model.classifier[-1].in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif hasattr(model, 'head'):                          # ConvNeXt (if used)
        in_features = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model):
    """Freeze everything except the final classifier head."""
    classifier_keywords = ['fc', 'classifier', 'head']
    for name, param in model.named_parameters():
        if not any(kw in name for kw in classifier_keywords):
            param.requires_grad = False
    return model


def unfreeze_last_block(model, model_name):
    """Unfreeze only the last block of the backbone (Scenario 2)."""
    last_block_keywords = {
        'resnet50':       'layer4',
        'densenet121':    'features.denseblock4',
        'efficientnet_b0': 'blocks.6',
    }
    keyword = last_block_keywords.get(model_name, '')
    for name, param in model.named_parameters():
        if keyword in name or any(kw in name for kw in ['fc', 'classifier', 'head']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def count_parameters(model):
    """Returns (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_efficiency_metrics(model, input_size=(1, 3, 224, 224), device='cpu'):
    """
    Compute and print Parameters, MACs, FLOPs.
    Called at start of every training/evaluation script.
    """
    model.eval().to(device)
    dummy = torch.randn(input_size).to(device)

    # MACs via thop
    macs, params = profile(model, inputs=(dummy,), verbose=False)

    # FLOPs via fvcore
    flops_analysis = FlopCountAnalysis(model, dummy)
    flops = flops_analysis.total()

    print(f"\n{'='*45}")
    print(f"  Efficiency Metrics")
    print(f"  Parameters : {params/1e6:.2f} M")
    print(f"  MACs       : {macs/1e9:.2f} G")
    print(f"  FLOPs      : {flops/1e9:.2f} G")
    print(f"{'='*45}\n")

    return {
        'params_M': round(params / 1e6, 2),
        'macs_G':   round(macs   / 1e9, 2),
        'flops_G':  round(flops  / 1e9, 2),
    }


def get_layer_names(model_name):
    """
    Documented layer selection for Scenario 5.
    Returns dict with early / middle / final layer names
    that can be used with register_forward_hook.
    """
    layer_map = {
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
    return layer_map.get(model_name, {})