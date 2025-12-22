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


'''
🎯 Perché 768 è un buon compromesso:
- È vicino alla mediana, quindi conserva almeno metà delle sequenze quasi complete.
- È una potenza di 2 → batching e GPU efficiency.
- Rimane gestibile in termini di VRAM, mentre 1024 potrebbe essere troppo.
- Troncamenti drastici (es. con T=300) dimezzavano la sequenza media; ora riduci solo il 5–10% in media.
'''


class FLAG3DDataset(Dataset):
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

    def __len__(self):
        return len(self.df)

    def _process_text(self, action_id):
        info = self.annotations[str(action_id)]
        fields = ['name', 'description', 'breathing', 'key_points', 'mistakes', 'notes']
        full_text = " ".join(info.get(f, '') for f in fields if info.get(f))
        return full_text

    def _process_pose(self, keypoint_array):
        """
        keypoint_array: (T, V, 3) come nel dataset originale FLAG3D.
        Output finale per 2S-AGCN:
            (T, V, 3) → nel DataLoader diventa (B, T, V, C).
        Il model.forward() si aspetta (B, T, V, C) e permuta internamente.
        """
        # Vecchio comportamento: squeeze su prima dim se presente
        keypoints = keypoint_array.squeeze(0)  # (T, V, 3) di solito (T, 25, 3)
        T, V, C = keypoints.shape
        assert C == 3, f"Expected last dim = 3, got {C}"

        # Padding / trimming su T (come prima)
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

        # Converte in tensor
        keypoints = torch.tensor(keypoints, dtype=torch.float32)  # (T, V, 3)

        # 🔁 Augmentazione opzionale: lavoriamo in (B, T, V, 3) per compatibilità con pose_augmentation
        if self.augment and random.random() < self.augment_prob:
            # Aggiungi batch fittizio
            keypoints = keypoints.unsqueeze(0)  # (1, T, V, 3)
            keypoints = apply_pose_augmentation(keypoints)  # (1, T, V, 3) invariato in shape
            keypoints = keypoints.squeeze(0)  # Torna a (T, V, 3)

        # Output finale per 2S-AGCN: (T, V, 3) - nessuna permutazione necessaria
        # Il model.forward() si aspetta (B, T, V, C) e permuta internamente a (B, C, T, V)
        return keypoints

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Pose
        keypoint_seq = self.keypoints['annotations'][idx]['keypoint']
        keypoints = self._process_pose(keypoint_seq)  # (V, 3, T)

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
            'pose': keypoints.to(self.device),  # (T, 25, 3) → batched: (B, T, V, C)
            'input_ids': tokenized['input_ids'].squeeze(0).to(self.device),
            'attention_mask': tokenized['attention_mask'].squeeze(0).to(self.device),
            'label': row['label'],
        }
