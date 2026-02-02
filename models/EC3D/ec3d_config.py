"""
Configurazione condivisa per gli esperimenti EC3D.

Questo modulo fornisce:
- Costanti di configurazione per gli esperimenti
- Dataset class ottimizzata per dati resampled e legacy
- Funzioni helper per caricare i dati

Uso nei notebook:
    from ec3d_config import (
        USE_RESAMPLED, TARGET_FRAMES, 
        EC3DDataset, load_ec3d_data
    )
"""

import sys
from pathlib import Path
import pickle
import json

import numpy as np
import torch
from torch.utils.data import Dataset

# ==============================================================================
# CONFIGURAZIONE GLOBALE (modificare qui per cambiare comportamento)
# ==============================================================================

# Flag principale: se True, usa i dati pre-resampled
# Se False, usa il comportamento legacy (padding/truncation)
USE_RESAMPLED = False  # Default: OFF per retrocompatibilità

# Lunghezza target per i dati resampled
TARGET_FRAMES = 300

# Lunghezza legacy per padding/truncation (usata se USE_RESAMPLED=False)
LEGACY_MAX_LEN = 150

# ==============================================================================
# PATH SETUP
# ==============================================================================

# Root del progetto
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent.parent

# Aggiungi root al path per importare utils
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils import load_ec3d, DEFAULT_TARGET_FRAMES


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class EC3DDataset(Dataset):
    """
    Dataset PyTorch per EC3D con supporto per dati resampled e legacy.
    
    Modalità:
    1. Resampled (use_resampled=True): 
       - Sequenze già (target_frames, 25, 3) 
       - Nessun preprocessing aggiuntivo
       
    2. Legacy (use_resampled=False):
       - Sequenze originali (T, 3, 25) con T variabile
       - Trasposizione a (T, 25, 3)
       - Padding con zeri o uniform sampling se T > max_len
    
    Args:
        sequences: Lista di sequenze numpy
        labels: Array di labels
        indices: Indici delle sequenze da usare
        use_resampled: Se True, assume sequenze già resampled
        max_len: Lunghezza massima per modalità legacy
    """
    
    def __init__(
        self, 
        sequences, 
        labels, 
        indices,
        use_resampled: bool = False,
        max_len: int = None
    ):
        self.sequences = [sequences[i] for i in indices]
        self.labels = labels[indices]
        self.use_resampled = use_resampled
        
        if use_resampled:
            # Sequenze già resampled: nessun max_len necessario
            # Assumiamo che abbiano tutte la stessa lunghezza
            self.max_len = sequences[indices[0]].shape[0]
        else:
            # Modalità legacy: calcola max_len se non specificato
            if max_len is None:
                self.max_len = max(seq.shape[0] for seq in self.sequences)
            else:
                self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.use_resampled:
            return self._get_resampled(idx)
        else:
            return self._get_legacy(idx)
    
    def _get_resampled(self, idx):
        """Modalità resampled: sequenze già (T, 25, 3)."""
        seq = self.sequences[idx]  # Già (target_frames, 25, 3)
        label = self.labels[idx]
        
        # Verifica shape
        assert seq.ndim == 3 and seq.shape[1] == 25 and seq.shape[2] == 3, \
            f"Shape inattesa per dati resampled: {seq.shape}"
        
        seq = torch.from_numpy(seq.astype(np.float32))
        label = torch.tensor(label, dtype=torch.long)
        
        return seq, label
    
    def _get_legacy(self, idx):
        """Modalità legacy: sequenze originali (T, 3, 25)."""
        seq = self.sequences[idx]  # (T, 3, 25)
        label = self.labels[idx]
        
        # Trasposizione: (T, 3, 25) -> (T, 25, 3)
        seq = np.transpose(seq, (0, 2, 1))  # (T, 25, 3)
        
        T, V, C = seq.shape
        
        # Padding/Truncation a max_len
        if T < self.max_len:
            pad = np.zeros((self.max_len - T, V, C), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        elif T > self.max_len:
            # Uniform sampling
            indices = np.linspace(0, T - 1, self.max_len).astype(int)
            seq = seq[indices]
        
        seq = torch.from_numpy(seq.astype(np.float32))
        label = torch.tensor(label, dtype=torch.long)
        
        return seq, label
    
    @property
    def output_frames(self) -> int:
        """Restituisce la lunghezza temporale delle sequenze."""
        return self.max_len


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_ec3d_data(
    use_resampled: bool = None,
    target_frames: int = None,
    no_unknown: bool = False,
    data_dir: Path = None
):
    """
    Carica i dati EC3D con la configurazione appropriata.
    
    Args:
        use_resampled: Se None, usa il valore globale USE_RESAMPLED
        target_frames: Se None, usa il valore globale TARGET_FRAMES
        no_unknown: Se True, carica versione senza classe Unknown
        data_dir: Directory dei dati (default: data/EC3D)
    
    Returns:
        dict con 'sequences', 'labels', 'meta', 'train_indices', 'test_indices', etc.
    """
    # Usa valori globali se non specificati
    if use_resampled is None:
        use_resampled = USE_RESAMPLED
    if target_frames is None:
        target_frames = TARGET_FRAMES
    if data_dir is None:
        data_dir = _project_root / "data" / "EC3D"
    
    return load_ec3d(
        data_dir=data_dir,
        no_unknown=no_unknown,
        return_split=True,
        resampled=use_resampled,
        target_frames=target_frames
    )


def get_max_len(use_resampled: bool = None, target_frames: int = None) -> int:
    """
    Restituisce la lunghezza da usare per i batch.
    
    Args:
        use_resampled: Se None, usa il valore globale
        target_frames: Se None, usa il valore globale
    
    Returns:
        int: TARGET_FRAMES se resampled, LEGACY_MAX_LEN altrimenti
    """
    if use_resampled is None:
        use_resampled = USE_RESAMPLED
    if target_frames is None:
        target_frames = TARGET_FRAMES
    
    return target_frames if use_resampled else LEGACY_MAX_LEN


def print_config():
    """Stampa la configurazione corrente."""
    print("=" * 50)
    print("EC3D CONFIGURATION")
    print("=" * 50)
    print(f"USE_RESAMPLED:    {USE_RESAMPLED}")
    print(f"TARGET_FRAMES:    {TARGET_FRAMES}")
    print(f"LEGACY_MAX_LEN:   {LEGACY_MAX_LEN}")
    print(f"Project root:     {_project_root}")
    print("=" * 50)


# ==============================================================================
# LABEL MAPPINGS (per comodità)
# ==============================================================================

ID_TO_NAME = {
    0: "SQUAT - Correct",
    1: "SQUAT - Feets too wide",
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

ID_TO_NAME_NO_UNKNOWN = {
    0: "SQUAT - Correct",
    1: "SQUAT - Feets too wide",
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

SHORT_NAMES = {
    0: "SQ-OK", 1: "SQ-Wide", 2: "SQ-Knee", 3: "SQ-Shallow",
    4: "SQ-Lean", 5: "SQ-Unk", 6: "LU-OK", 7: "LU-Shallow",
    8: "LU-Knee", 9: "PL-OK", 10: "PL-Banana", 11: "PL-Roll"
}


if __name__ == "__main__":
    # Test del modulo
    print_config()
    
    print("\n🧪 Test caricamento dati...")
    try:
        data = load_ec3d_data(use_resampled=False, no_unknown=False)
        print(f"✅ Dati legacy caricati: {len(data['sequences'])} sequenze")
        print(f"   Shape esempio: {data['sequences'][0].shape}")
    except Exception as e:
        print(f"⚠️ Dati legacy: {e}")
    
    # Test resampled (potrebbe non esistere ancora)
    try:
        data_res = load_ec3d_data(use_resampled=True, no_unknown=False)
        print(f"✅ Dati resampled caricati: {len(data_res['sequences'])} sequenze")
        print(f"   Shape esempio: {data_res['sequences'][0].shape}")
    except Exception as e:
        print(f"⚠️ Dati resampled non disponibili: {e}")
        print("   Esegui: python data/EC3D/build_ec3d_resampled.py")



