from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import DistilBertTokenizerFast

import os
import sys
from pathlib import Path
import random

# Aggiunge la root del progetto al PYTHONPATH
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.FLAG3D.pose_augmentation import apply_pose_augmentation
from utils import resample_sequence, DEFAULT_TARGET_FRAMES


'''
🎯 Perché 768  (era) un buon compromesso:
- È vicino alla mediana, quindi conserva almeno metà delle sequenze quasi complete.
- È una potenza di 2 → batching e GPU efficiency.
- Rimane gestibile in termini di VRAM, mentre 1024 potrebbe essere troppo.
- Troncamenti drastici (es. con T=300) dimezzavano la sequenza media; ora riduci solo il 5–10% in media.

🎯 Configurazione temporale FLAG3D con augmentation:

Legacy (default, use_resampling=False):
- max_frames=300 per compatibilità con codice esistente
- Padding con zeri o truncation semplice

Resampling (opt-in, use_resampling=True):
- target_frames=300 (parametrico) 
- Resampling uniforme invece di padding/truncation
- Garantisce coerenza temporale con EC3D downstream

🆕 Nota: l'augmentation offline dovrebbe usare lo stesso target_frames
    per garantire coerenza. Vedi augment_and_save_FLAG3D.py
'''


class FLAG3DDataset(Dataset):
    """
    PyTorch Dataset per FLAG3D con supporto per augmentation online e resampling opt-in.
    
    Modalità di preprocessing temporale:
    1. Legacy (default): padding con zeri o truncation a max_frames
    2. Resampling (opt-in): resampling uniforme a target_frames
    
    Args:
        metadata_df: DataFrame con metadati delle sequenze
        annotations_dict: Dizionario con annotazioni testuali per action_id
        keypoints_data: Dizionario con dati keypoint
        tokenizer: Tokenizer per il testo (default: DistilBertTokenizerFast)
        max_frames: Lunghezza massima per modalità legacy (default: 300)
        text_mode: Modalità testo ('full' usa tutti i campi)
        device: Device per i tensori
        augment: Se True, applica augmentation online
        augment_prob: Probabilità di applicare augmentation (default: 0.5)
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
        max_frames=300,
        text_mode='full',
        device='cpu',
        augment=False,
        augment_prob=0.5,
        # Nuovi parametri per resampling opt-in
        use_resampling: bool = False,
        target_frames: int = DEFAULT_TARGET_FRAMES,
        resample_method: str = "linear"
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.annotations = annotations_dict
        self.keypoints = keypoints_data
        self.tokenizer = tokenizer or DistilBertTokenizerFast.from_pretrained(
            'distilbert-base-uncased'
        )
        self.max_frames = max_frames
        self.text_mode = text_mode
        self.device = device
        self.augment = augment
        self.augment_prob = augment_prob
        
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
        keypoints = keypoint_array.squeeze(0)  # (T, V, 3) di solito (T, 25, 3)
        T, V, C = keypoints.shape
        assert C == 3, f"Expected last dim = 3, got {C}"

        if T > self.max_frames:
            keypoints = keypoints[:self.max_frames]  # (max_frames, V, 3)
            T = self.max_frames
        elif T < self.max_frames:
            pad_len = self.max_frames - T
            padding = np.zeros(
                (pad_len, keypoints.shape[1], keypoints.shape[2]),
                dtype=keypoints.dtype,
            )
            keypoints = np.concatenate([keypoints, padding], axis=0)  # (max_frames, V, 3)
            T = self.max_frames

        keypoints = torch.tensor(keypoints, dtype=torch.float32)  # (T, V, 3)
        return keypoints

    def _process_pose_resampled(self, keypoint_array):
        """
        Preprocessing con resampling uniforme.
        
        Input: (1, T, 25, 3) o (T, 25, 3)
        Output: (target_frames, 25, 3) tensor float32
        """
        keypoints = keypoint_array.squeeze(0)  # (T, V, 3)
        keypoints = np.asarray(keypoints, dtype=np.float32)
        
        # Applica resampling uniforme
        keypoints = resample_sequence(
            keypoints, 
            target_len=self.target_frames, 
            method=self.resample_method
        )
        
        keypoints = torch.tensor(keypoints, dtype=torch.float32)  # (target_frames, V, 3)
        return keypoints

    def _process_pose(self, keypoint_array):
        """
        Processa una sequenza di pose con preprocessing temporale e augmentation opzionale.
        
        Output: (T*, V, 3) dove T* = target_frames o max_frames
        """
        # Scegli preprocessing temporale
        if self.use_resampling:
            keypoints = self._process_pose_resampled(keypoint_array)
        else:
            keypoints = self._process_pose_legacy(keypoint_array)

        # 🔁 Augmentazione opzionale: lavoriamo in (B, T, V, 3) per compatibilità
        if self.augment and random.random() < self.augment_prob:
            keypoints = keypoints.unsqueeze(0)  # (1, T, V, 3)
            keypoints = apply_pose_augmentation(keypoints)  # (1, T, V, 3)
            keypoints = keypoints.squeeze(0)  # (T, V, 3)

        return keypoints

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Pose
        keypoint_seq = self.keypoints['annotations'][idx]['keypoint']
        keypoints = self._process_pose(keypoint_seq)  # (T*, V, 3)

        # Testo
        text = self._process_text(row['action_id'])
        tokenized = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt',
        )

        return {
            'pose': keypoints.to(self.device),  # (T*, 25, 3) → batched: (B, T*, V, C)
            'input_ids': tokenized['input_ids'].squeeze(0).to(self.device),
            'attention_mask': tokenized['attention_mask'].squeeze(0).to(self.device),
            'label': row['label'],
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
            'augment': self.augment,
            'augment_prob': self.augment_prob if self.augment else None,
            'text_mode': self.text_mode,
            'num_samples': len(self.df),
        }
