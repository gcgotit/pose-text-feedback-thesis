from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import DistilBertTokenizerFast

'''
🎯 Perché 768 è un buon compromesso:

È vicino alla mediana, quindi conserva almeno metà delle sequenze quasi complete.

È una potenza di 2 → batching e GPU efficiency.

Rimane gestibile in termini di VRAM, mentre 1024 potrebbe essere troppo.

Troncamenti drastici (es. con T=300) dimezzavano la sequenza media; ora riduci solo il 5–10% in media.
'''

class FLAG3DDataset(Dataset):
    def __init__(self, metadata_df, annotations_dict, keypoints_data, tokenizer=None, 
                 max_frames=550, text_mode='full', device='cpu'):
        self.df = metadata_df.reset_index(drop=True)
        self.annotations = annotations_dict
        self.keypoints = keypoints_data
        self.tokenizer = tokenizer or DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.max_frames = max_frames
        self.text_mode = text_mode
        self.device = device

    def __len__(self):
        return len(self.df)

    def _process_text(self, action_id):
        info = self.annotations[str(action_id)]
        fields = ['name', 'description', 'breathing', 'key_points', 'mistakes', 'notes']
        full_text = " ".join(info.get(f, '') for f in fields if info.get(f))
        return full_text

    def _process_pose(self, keypoint_array):
        keypoints = keypoint_array.squeeze(0)  # (T, 25, 3)
        T = keypoints.shape[0]
        if T > self.max_frames:
            keypoints = keypoints[:self.max_frames]
        elif T < self.max_frames:
            pad_len = self.max_frames - T
            padding = np.zeros((pad_len, keypoints.shape[1], keypoints.shape[2]), dtype=keypoints.dtype)
            keypoints = np.concatenate([keypoints, padding], axis=0)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)  # (T, 25, 3)
        keypoints = keypoints.permute(2, 0, 1).permute(1, 2, 0)  # → (T, 25, 3)
        return keypoints

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
            'pose': keypoints.to(self.device),              # (25, 3, T)
            'input_ids': tokenized['input_ids'].squeeze(0).to(self.device),
            'attention_mask': tokenized['attention_mask'].squeeze(0).to(self.device),
            'label': row['label']
        }