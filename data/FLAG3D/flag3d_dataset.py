from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import DistilBertTokenizerFast

import sys
from pathlib import Path

# Aggiunge la root del progetto al PYTHONPATH per importare utils
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import resample_sequence, DEFAULT_TARGET_FRAMES

'''
🎯 Perché 550 è un buon compromesso per FLAG3D:

È vicino alla mediana, quindi conserva almeno metà delle sequenze quasi complete.

Rimane gestibile in termini di VRAM, mentre 1024 potrebbe essere troppo.

Troncamenti drastici (es. con T=300) dimezzavano la sequenza media; ora riduci solo il 5–10% in media.

🆕 Nuova modalità resampling (opt-in):
- Invece di padding/truncation, usa resampling uniforme a target_frames
- Garantisce coerenza temporale tra FLAG3D training e EC3D downstream
- Attivabile con use_resampling=True
'''


class FLAG3DDataset(Dataset):
    """
    PyTorch Dataset per FLAG3D con supporto per due modalità di preprocessing temporale:
    
    1. Legacy (default): padding con zeri o truncation semplice
    2. Resampling (opt-in): resampling uniforme a lunghezza fissa
    
    Args:
        metadata_df: DataFrame con metadati delle sequenze
        annotations_dict: Dizionario con annotazioni testuali per action_id
        keypoints_data: Dizionario con dati keypoint (da flag3d_keypoint.pkl)
        tokenizer: Tokenizer per il testo (default: DistilBertTokenizerFast)
        max_frames: Lunghezza massima per modalità legacy (default: 550)
        text_mode: Modalità testo ('full' usa tutti i campi)
        device: Device per i tensori ('cpu' o 'cuda')
        use_resampling: Se True, usa resampling invece di padding/truncation
        target_frames: Lunghezza target per resampling (default: 300)
        resample_method: Metodo di resampling ('linear' o 'index')
    """
    
    def __init__(
        self, 
        metadata_df, 
        annotations_dict, 
        keypoints_data, 
        tokenizer=None, 
        max_frames=550, 
        text_mode='full', 
        device='cpu',
        # Nuovi parametri per resampling opt-in
        use_resampling: bool = False,
        target_frames: int = DEFAULT_TARGET_FRAMES,
        resample_method: str = "linear"
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.annotations = annotations_dict
        self.keypoints = keypoints_data
        self.tokenizer = tokenizer or DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.max_frames = max_frames
        self.text_mode = text_mode
        self.device = device
        
        # Resampling configuration
        self.use_resampling = use_resampling
        self.target_frames = target_frames
        self.resample_method = resample_method
        
        # Determina la lunghezza output effettiva
        self._output_frames = target_frames if use_resampling else max_frames

    def __len__(self):
        return len(self.df)

    def _process_text(self, action_id):
        """Processa le annotazioni testuali concatenando tutti i campi disponibili."""
        info = self.annotations[str(action_id)]
        fields = ['name', 'description', 'breathing', 'key_points', 'mistakes', 'notes']
        full_text = " ".join(info.get(f, '') for f in fields if info.get(f))
        return full_text

    def _process_pose_legacy(self, keypoint_array):
        """
        Preprocessing legacy: padding con zeri o truncation semplice.
        
        Input: (1, T, 25, 3) o (T, 25, 3)
        Output: (max_frames, 25, 3) tensor float32
        """
        keypoints = keypoint_array.squeeze(0)  # (T, 25, 3)
        T = keypoints.shape[0]
        
        if T > self.max_frames:
            # Truncation: prendi i primi max_frames
            keypoints = keypoints[:self.max_frames]
        elif T < self.max_frames:
            # Padding: aggiungi zeri alla fine
            pad_len = self.max_frames - T
            padding = np.zeros((pad_len, keypoints.shape[1], keypoints.shape[2]), dtype=keypoints.dtype)
            keypoints = np.concatenate([keypoints, padding], axis=0)
        
        keypoints = torch.tensor(keypoints, dtype=torch.float32)  # (max_frames, 25, 3)
        return keypoints

    def _process_pose_resampled(self, keypoint_array):
        """
        Preprocessing con resampling uniforme.
        
        Input: (1, T, 25, 3) o (T, 25, 3)
        Output: (target_frames, 25, 3) tensor float32
        """
        keypoints = keypoint_array.squeeze(0)  # (T, 25, 3)
        keypoints = np.asarray(keypoints, dtype=np.float32)
        
        # Applica resampling uniforme
        keypoints = resample_sequence(
            keypoints, 
            target_len=self.target_frames, 
            method=self.resample_method
        )
        
        keypoints = torch.tensor(keypoints, dtype=torch.float32)  # (target_frames, 25, 3)
        return keypoints

    def _process_pose(self, keypoint_array):
        """Dispatcher per il preprocessing delle pose."""
        if self.use_resampling:
            return self._process_pose_resampled(keypoint_array)
        else:
            return self._process_pose_legacy(keypoint_array)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        keypoint_seq = self.keypoints['annotations'][idx]['keypoint']
        keypoints = self._process_pose(keypoint_seq)
        text = self._process_text(row['action_id'])

        tokenized = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return {
            'pose': keypoints.to(self.device),              # (T*, 25, 3) dove T* = target_frames o max_frames
            'input_ids': tokenized['input_ids'].squeeze(0).to(self.device),
            'attention_mask': tokenized['attention_mask'].squeeze(0).to(self.device),
            'label': row['label']
        }
    
    @property
    def output_frames(self) -> int:
        """Restituisce la lunghezza temporale delle sequenze in output."""
        return self._output_frames
    
    def get_config(self) -> dict:
        """Restituisce la configurazione corrente del dataset."""
        return {
            'use_resampling': self.use_resampling,
            'target_frames': self.target_frames if self.use_resampling else None,
            'max_frames': self.max_frames if not self.use_resampling else None,
            'resample_method': self.resample_method if self.use_resampling else None,
            'output_frames': self._output_frames,
            'text_mode': self.text_mode,
            'num_samples': len(self.df),
        }
