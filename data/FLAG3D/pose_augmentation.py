import torch
import random
import math


def jitter_pose(pose, std=0.01):
    """Aggiunge rumore gaussiano alle coordinate (x, y, z).
    pose: (B, T, V, 3)
    """
    noise = torch.randn_like(pose) * std
    return pose + noise


def scale_pose(pose, scale_range=(0.95, 1.05)):
    """Scala la posa uniformemente.
    pose: (B, T, V, 3)
    """
    scale = random.uniform(*scale_range)
    return pose * scale


def rotate_pose(pose, max_angle_deg=10):
    """Ruota la posa attorno all'asse Y (verticale).
    pose: (B, T, V, 3)
    """
    assert pose.ndim == 4 and pose.shape[-1] == 3, \
        f"rotate_pose si aspetta (B, T, V, 3), got {pose.shape}"

    B, T, V, C = pose.shape
    assert C == 3, f"Ultima dimensione deve essere 3, trovata {C}"

    angle_rad = math.radians(random.uniform(-max_angle_deg, max_angle_deg))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotation_matrix = torch.tensor(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=pose.dtype,
        device=pose.device,
    )  # (3, 3)

    # (B, T, V, 3) → (B*T*V, 3)
    pose_flat = pose.view(-1, 3)
    rotated_flat = pose_flat @ rotation_matrix.T  # (B*T*V, 3)

    rotated = rotated_flat.view(B, T, V, 3)  # (B, T, V, 3)
    return rotated


def apply_pose_augmentation(pose, num_transforms=2):
    """Applica una combinazione casuale di trasformazioni.
    Input: pose (B, T, V, 3)
    Output: stessa shape
    """
    assert pose.ndim == 4 and pose.shape[-1] == 3, \
        f"Expected input shape (B, T, V, 3), got {pose.shape}"

    transforms = [jitter_pose, scale_pose, rotate_pose]
    num_transforms = min(num_transforms, len(transforms))
    selected = random.sample(transforms, k=num_transforms)

    for tf in selected:
        pose = tf(pose)

    return pose
