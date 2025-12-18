# train_dual_encoder.py

from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import sys
from pathlib import Path

# Aggiunge la root del progetto al PYTHONPATH
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from torch.utils.data import Subset
from models.pose_encoder.twostream_agcn import TwoStreamAGCN
from models.text_encoder.distilbert_adapter import DistilBERTTextEncoder
from data.FLAG3D.flag3d_dataset import FLAG3DDataset
from utils import ntxent_loss  # contrastive loss
import json
import pickle

# Percorso alla directory dei dati
data_dir = project_root / "data" / "FLAG3D"

# 1. Carica il DataFrame dei metadati
metadata_df = pd.read_csv(data_dir / "flag3d_metadata.csv")

# 2. Carica le annotazioni testuali
with open(data_dir / "flag3d_annotations.json", "r") as f:
    annotations_dict = json.load(f)

# 3. Carica le pose/keypoint
with open(data_dir / "flag3d_keypoint.pkl", "rb") as f:
    keypoints_data = pickle.load(f)

with open(data_dir / "flag3d_split.json") as f:
    split = json.load(f)
train_indices = split["train_indices"]
val_indices = split["val_indices"]


 # Costruisci il dataset                                                                                                  
full_dataset = FLAG3DDataset(
    metadata_df=metadata_df,
    annotations_dict=annotations_dict,
    keypoints_data=keypoints_data,
    device='cuda'
)

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# printami le dimensioni dei dataset per verificare se gli indici siano corretti rispetto a data_dir / "flag3d_split.json" che contiene gli indici corretti dello split train / val
print(len(train_dataset))
print(len(val_dataset))

# Controlla se ci sono indici in comune tra train e val (campioni uguali)
overlap = set(train_indices).intersection(val_indices)
print(f"Numero di campioni in comune tra train e val: {len(overlap)}")
if overlap:
    print("Esempio di indici in comune:", list(overlap)[:10])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# printami le dimensioni dei dataloader per verificare se gli indici siano corretti rispetto a data_dir / "flag3d_split.json" che contiene gli indici corretti dello split train / val
print(len(train_loader))
print(len(val_loader))



