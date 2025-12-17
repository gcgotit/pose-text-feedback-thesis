import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Aggiunge la root del progetto al PYTHONPATH
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.pose_encoder.twostream_agcn import TwoStreamAGCN
from models.text_encoder.distilbert_adapter import DistilBERTTextEncoder

class DualEncoder(nn.Module):
    def __init__(self, pose_in_channels=3, pose_hidden_dim=64, pose_output_dim=128,
                 text_output_dim=128, temperature=0.07):
        super(DualEncoder, self).__init__()
        self.pose_encoder = TwoStreamAGCN(
            in_channels=pose_in_channels,
            hidden_dim=pose_hidden_dim,
            output_dim=pose_output_dim
        )
        self.text_encoder = DistilBERTTextEncoder(output_dim=text_output_dim)
        self.temperature = temperature

    def forward(self, pose_input, text_input):
        # pose_input: (B, C, T, V)
        # text_input: list[str] of length B

        pose_embed = self.pose_encoder(pose_input)   # (B, D)
        text_embed = self.text_encoder(text_input)   # (B, D)

        # Normalize embeddings
        pose_embed = F.normalize(pose_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)

        return pose_embed, text_embed

    def compute_contrastive_loss(self, pose_embed, text_embed):
        # InfoNCE loss
        logits = torch.matmul(pose_embed, text_embed.T) / self.temperature
        labels = torch.arange(pose_embed.size(0)).to(pose_embed.device)
        loss_i2t = F.cross_entropy(logits, labels)     # pose -> text
        loss_t2i = F.cross_entropy(logits.T, labels)   # text -> pose
        return (loss_i2t + loss_t2i) / 2
