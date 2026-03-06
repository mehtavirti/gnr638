import torch
import numpy as np
from torchvision import transforms
from scipy.ndimage import uniform_filter


def apply_gaussian_noise(tensor, sigma):
    noise = torch.randn_like(tensor) * sigma
    return torch.clamp(tensor + noise, 0.0, 1.0)


def apply_motion_blur(tensor, kernel_size=15):
    """Horizontal motion blur via uniform filter."""
    np_img = tensor.permute(1, 2, 0).numpy()
    blurred = np.stack([
        uniform_filter(np_img[:, :, c], size=(1, kernel_size))
        for c in range(3)
    ], axis=2)
    return torch.from_numpy(blurred).permute(2, 0, 1).float()


def apply_brightness_shift(tensor, delta=0.2):
    return torch.clamp(tensor + delta, 0.0, 1.0)


def get_corruption_transform(corruption_type, severity=None):
    """
    Returns a callable that applies corruption to a tensor batch.
    corruption_type: 'gaussian' | 'motion_blur' | 'brightness'
    """
    if corruption_type == 'gaussian':
        return lambda x: apply_gaussian_noise(x, sigma=severity)
    elif corruption_type == 'motion_blur':
        return lambda x: torch.stack([apply_motion_blur(img) for img in x])
    elif corruption_type == 'brightness':
        return lambda x: apply_brightness_shift(x, delta=severity or 0.2)
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")