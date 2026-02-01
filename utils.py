import torch
import torch.nn.functional as F
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# ==============================================================================
# Configuration Constants
# ==============================================================================

# Default target frames for temporal resampling
DEFAULT_TARGET_FRAMES = 300


# ==============================================================================
# Temporal Resampling Utilities
# ==============================================================================

def resample_sequence(
    seq: np.ndarray, 
    target_len: int, 
    method: str = "linear"
) -> np.ndarray:
    """
    Resample a pose sequence to a fixed target length.
    
    This function handles both downsampling (T > target_len) and upsampling (T < target_len)
    using either linear interpolation or index-based resampling.
    
    Args:
        seq: Input sequence of shape (T, V, C) where:
             - T = number of frames (variable)
             - V = number of joints (25 for NTU-RGB+D)
             - C = coordinates (3 for x, y, z)
        target_len: Desired output length
        method: Resampling method:
                - "linear": Linear interpolation (recommended for upsampling)
                - "index": Index-based uniform sampling (faster, good for downsampling)
    
    Returns:
        Resampled sequence of shape (target_len, V, C) with dtype float32
    
    Example:
        >>> seq = np.random.randn(80, 25, 3).astype(np.float32)  # EC3D-like
        >>> resampled = resample_sequence(seq, target_len=300, method="linear")
        >>> print(resampled.shape)  # (300, 25, 3)
    """
    seq = np.asarray(seq, dtype=np.float32)
    
    if seq.ndim != 3:
        raise ValueError(f"Expected 3D array (T, V, C), got shape {seq.shape}")
    
    T, V, C = seq.shape
    
    # No resampling needed
    if T == target_len:
        return seq
    
    # Downsampling: use uniform index selection (fast and preserves key frames)
    if T > target_len:
        indices = np.linspace(0, T - 1, target_len).astype(int)
        return seq[indices].astype(np.float32)
    
    # Upsampling: T < target_len
    if method == "index":
        # Index-based: repeat frames uniformly
        indices = np.linspace(0, T - 1, target_len).astype(int)
        return seq[indices].astype(np.float32)
    
    elif method == "linear":
        # Linear interpolation for smooth upsampling
        # Interpolate each joint and coordinate independently
        t_original = np.linspace(0, 1, T)
        t_target = np.linspace(0, 1, target_len)
        
        resampled = np.zeros((target_len, V, C), dtype=np.float32)
        
        for v in range(V):
            for c in range(C):
                # Use numpy's interp for 1D linear interpolation
                resampled[:, v, c] = np.interp(t_target, t_original, seq[:, v, c])
        
        return resampled
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'index'.")


def resample_batch(
    sequences: List[np.ndarray],
    target_len: int,
    method: str = "linear",
    show_progress: bool = False
) -> List[np.ndarray]:
    """
    Resample a list of sequences to a fixed target length.
    
    Args:
        sequences: List of sequences, each with shape (T_i, V, C)
        target_len: Desired output length for all sequences
        method: Resampling method ("linear" or "index")
        show_progress: If True, show progress bar (requires tqdm)
    
    Returns:
        List of resampled sequences, each with shape (target_len, V, C)
    """
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(sequences, desc=f"Resampling to T={target_len}")
        except ImportError:
            iterator = sequences
    else:
        iterator = sequences
    
    return [resample_sequence(seq, target_len, method) for seq in iterator]


def get_sequence_length_stats(sequences: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute statistics on sequence lengths.
    
    Args:
        sequences: List of sequences, each with shape (T_i, ...)
    
    Returns:
        Dict with min, max, mean, std, median of sequence lengths
    """
    lengths = [seq.shape[0] for seq in sequences]
    return {
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        'mean': float(np.mean(lengths)),
        'std': float(np.std(lengths)),
        'median': float(np.median(lengths)),
        'count': len(lengths)
    }


# ==============================================================================
# EC3D Dataset Utilities
# ==============================================================================

# Label mapping for EC3D (WITH Unknown class - 12 classes)
EC3D_ID_TO_NAME_WITH_UNKNOWN = {
    0: "SQUAT - Correct",
    1: "SQUAT - Feet too wide",
    2: "SQUAT - Knees inward",
    3: "SQUAT - Not low enough",
    4: "SQUAT - Front bended",
    5: "SQUAT - Unknown",
    6: "LUNGES - Correct",
    7: "LUNGES - Not low enough",
    8: "LUNGES - Knees pass toes",
    9: "PLANK - Correct",
    10: "PLANK - Banana back",
    11: "PLANK - Rolled back",
}

# Label mapping for EC3D (NO Unknown class - 11 classes, paper-aligned)
EC3D_ID_TO_NAME_NO_UNKNOWN = {
    0: "SQUAT - Correct",
    1: "SQUAT - Feet too wide",
    2: "SQUAT - Knees inward",
    3: "SQUAT - Not low enough",
    4: "SQUAT - Front bended",
    5: "LUNGES - Correct",
    6: "LUNGES - Not low enough",
    7: "LUNGES - Knees pass toes",
    8: "PLANK - Correct",
    9: "PLANK - Banana back",
    10: "PLANK - Rolled back",
}

# Short names for plots
EC3D_SHORT_NAMES_WITH_UNKNOWN = {
    0: "SQ-OK", 1: "SQ-Wide", 2: "SQ-Knee", 3: "SQ-Shallow",
    4: "SQ-Lean", 5: "SQ-Unk", 6: "LU-OK", 7: "LU-Shallow",
    8: "LU-Knee", 9: "PL-OK", 10: "PL-Banana", 11: "PL-Roll"
}

EC3D_SHORT_NAMES_NO_UNKNOWN = {
    0: "SQ-OK", 1: "SQ-Wide", 2: "SQ-Knee", 3: "SQ-Shallow",
    4: "SQ-Lean", 5: "LU-OK", 6: "LU-Shallow",
    7: "LU-Knee", 8: "PL-OK", 9: "PL-Banana", 10: "PL-Roll"
}

# Mapping from old labels (with unknown) to new labels (no unknown)
# Old 5 (SQUAT-Unknown) is removed, 6-11 shift down by 1
EC3D_OLD_TO_NEW_LABEL_MAP = {
    0: 0,  # SQUAT - Correct
    1: 1,  # SQUAT - Feet too wide
    2: 2,  # SQUAT - Knees inward
    3: 3,  # SQUAT - Not low enough
    4: 4,  # SQUAT - Front bended
    # 5: REMOVED (Unknown)
    6: 5,  # LUNGES - Correct
    7: 6,  # LUNGES - Not low enough
    8: 7,  # LUNGES - Knees pass toes
    9: 8,  # PLANK - Correct
    10: 9,  # PLANK - Banana back
    11: 10, # PLANK - Rolled back
}


def load_ec3d(
    data_dir: str | Path,
    no_unknown: bool = False,
    return_split: bool = True,
    resampled: bool = False,
    target_frames: int = DEFAULT_TARGET_FRAMES
) -> Dict[str, Any]:
    """
    Load EC3D dataset with or without the Unknown class.
    
    Args:
        data_dir: Path to data/EC3D directory
        no_unknown: If True, load the paper-aligned version without Unknown class (11 classes)
                   If False, load the original version with Unknown class (12 classes)
        return_split: If True, also load and return the cross-subject split
        resampled: If True, load the pre-resampled version of the dataset
                   (requires running build_ec3d_resampled.py first)
        target_frames: Target frame count for resampled data (used in filename)
    
    Returns:
        dict with keys:
            - 'sequences': list of np.array - shape depends on version:
                * Original: (T, 3, 25) with variable T
                * Resampled: (target_frames, 25, 3) with fixed T
            - 'labels': np.array of labels
            - 'meta': list of metadata dicts
            - 'num_classes': number of classes (11 or 12)
            - 'id_to_name': label id to name mapping
            - 'short_names': short names for plots
            - 'train_indices': (if return_split) training indices
            - 'test_indices': (if return_split) test indices
            - 'resampled': bool indicating if data is resampled
            - 'target_frames': int, the target frame count (if resampled)
    """
    data_dir = Path(data_dir)
    
    # Determine which file to load based on options
    if resampled:
        # Load pre-resampled dataset
        suffix = f"_T{target_frames}_resampled"
        if no_unknown:
            pkl_path = data_dir / f"ec3d_sequences_no_unknown{suffix}.pkl"
            split_suffix = f"_no_unknown{suffix}"
        else:
            pkl_path = data_dir / f"ec3d_sequences{suffix}.pkl"
            split_suffix = suffix
        
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Resampled file not found: {pkl_path}\n"
                "Please run: python data/EC3D/build_ec3d_resampled.py first."
            )
        
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        result = {
            'sequences': data['sequences'],  # Already (target_frames, 25, 3)
            'labels': data['labels'],
            'meta': data['meta'],
            'resampled': True,
            'target_frames': target_frames,
        }
        
        # Set class info
        if no_unknown:
            result.update({
                'num_classes': 11,
                'id_to_name': EC3D_ID_TO_NAME_NO_UNKNOWN,
                'short_names': EC3D_SHORT_NAMES_NO_UNKNOWN,
                'old_to_new_map': EC3D_OLD_TO_NEW_LABEL_MAP,
            })
        else:
            result.update({
                'num_classes': 12,
                'id_to_name': EC3D_ID_TO_NAME_WITH_UNKNOWN,
                'short_names': EC3D_SHORT_NAMES_WITH_UNKNOWN,
            })
        
        if return_split:
            # Resampled datasets use the same split as original
            if no_unknown:
                split_path = data_dir / "split_cross_subject_no_unknown.json"
            else:
                split_path = data_dir / "split_cross_subject.json"
            
            if split_path.exists():
                with open(split_path, "r") as f:
                    split = json.load(f)
                result['train_indices'] = np.array(split['train_indices'])
                result['test_indices'] = np.array(split['test_indices'])
            else:
                raise FileNotFoundError(f"Split file not found: {split_path}")
    
    else:
        # Load original dataset (legacy behavior)
        if no_unknown:
            pkl_path = data_dir / "ec3d_sequences_no_unknown.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(
                    f"File not found: {pkl_path}\n"
                    "Please run notebook 00_ec3d_build_no_unknown.ipynb first to create this file."
                )
            
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            result = {
                'sequences': data['sequences'],
                'labels': data['labels'],
                'meta': data['meta'],
                'num_classes': 11,
                'id_to_name': EC3D_ID_TO_NAME_NO_UNKNOWN,
                'short_names': EC3D_SHORT_NAMES_NO_UNKNOWN,
                'old_to_new_map': EC3D_OLD_TO_NEW_LABEL_MAP,
                'resampled': False,
            }
            
            if return_split:
                split_path = data_dir / "split_cross_subject_no_unknown.json"
                if split_path.exists():
                    with open(split_path, "r") as f:
                        split = json.load(f)
                    result['train_indices'] = np.array(split['train_indices'])
                    result['test_indices'] = np.array(split['test_indices'])
                else:
                    raise FileNotFoundError(
                        f"Split file not found: {split_path}\n"
                        "Please run notebook 00_ec3d_build_no_unknown.ipynb first."
                    )
        else:
            pkl_path = data_dir / "ec3d_sequences.pkl"
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            result = {
                'sequences': data['sequences'],
                'labels': data['labels'],
                'meta': data['meta'],
                'num_classes': 12,
                'id_to_name': EC3D_ID_TO_NAME_WITH_UNKNOWN,
                'short_names': EC3D_SHORT_NAMES_WITH_UNKNOWN,
                'resampled': False,
            }
            
            if return_split:
                split_path = data_dir / "split_cross_subject.json"
                with open(split_path, "r") as f:
                    split = json.load(f)
                result['train_indices'] = np.array(split['train_indices'])
                result['test_indices'] = np.array(split['test_indices'])
    
    return result


def get_ec3d_label_info(no_unknown: bool = False) -> Tuple[Dict, Dict, int]:
    """
    Get EC3D label information without loading data.
    
    Args:
        no_unknown: If True, return info for 11-class version
    
    Returns:
        Tuple of (id_to_name, short_names, num_classes)
    """
    if no_unknown:
        return EC3D_ID_TO_NAME_NO_UNKNOWN, EC3D_SHORT_NAMES_NO_UNKNOWN, 11
    else:
        return EC3D_ID_TO_NAME_WITH_UNKNOWN, EC3D_SHORT_NAMES_WITH_UNKNOWN, 12


# ==============================================================================
# Contrastive Learning Loss Functions  
# ==============================================================================

def ntxent_loss(embeddings1: torch.Tensor, embeddings2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Calcola la Normalized Temperature-scaled Cross Entropy Loss (NTXentLoss) tra due insiemi di embedding.
    
    Args:
        embeddings1 (torch.Tensor): Tensore degli embedding del primo dominio (es. pose), 
                                    dimensioni (N, D) dove N è il numero di campioni del batch e D la dimensionalità.
        embeddings2 (torch.Tensor): Tensore degli embedding del secondo dominio (es. testo),
                                    dimensioni (N, D), corrispondenti uno-a-uno con embeddings1.
        temperature (float, opzionale): Parametro di temperatura τ per scalare le similitudini (default = 0.5).
        
    Returns:
        torch.Tensor: Un tensore scalare (0-dimension) con il valore della loss contrastiva NTXent.
    """
    # Verifica che il numero di esempi corrisponda
    if embeddings1.shape[0] != embeddings2.shape[0]:
        raise ValueError("I tensori embeddings1 ed embeddings2 devono avere lo stesso numero di campioni.")
    
    # Assicura che i tensori siano sullo stesso dispositivo (CPU o GPU)
    if embeddings1.device != embeddings2.device:
        embeddings2 = embeddings2.to(embeddings1.device)
    
    # Normalizza gli embedding lungo la dimensione delle caratteristiche (D) per ottenere vettori unitari
    embeddings1_norm = F.normalize(embeddings1, p=2, dim=1)
    embeddings2_norm = F.normalize(embeddings2, p=2, dim=1)
    
    # Calcola la matrice di similarità (dot product) tra tutti gli embedding
    # Risulterà una matrice N x N dove entry (i,j) = sim(embeddings1[i], embeddings2[j])
    similarity_matrix = embeddings1_norm @ embeddings2_norm.T  # prodotto matrice (N,D)x(D,N) -> (N,N)
    
    # Crea il target per il calcolo della cross-entropy: per ogni riga i, il "label corretto" è i (coppia positiva)
    device = similarity_matrix.device
    N = similarity_matrix.size(0)
    target = torch.arange(N, device=device)
    
    # Calcola la loss cross-entropia per le due direzioni:
    # 1. Considerando embeddings1 come anchor e classificando embeddings2
    loss_i = F.cross_entropy(similarity_matrix / temperature, target, reduction='mean')
    # 2. Considerando embeddings2 come anchor e classificando embeddings1 
    loss_j = F.cross_entropy(similarity_matrix.T / temperature, target, reduction='mean')
    
    # Media delle due loss (posizioni positive considerate da entrambe le prospettive)
    loss = 0.5 * (loss_i + loss_j)
    return loss
