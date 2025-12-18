"""
train_dual_encoder.py
---------------------
Addestramento del dual encoder (2S-AGCN + DistilBERT) su FLAG3D.
Include:
- Train/val split con Subset
- Loss contrastiva NT-Xent
- Early stopping su validation loss
- Salvataggio log, pesi, grafici
"""

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
import matplotlib.pyplot as plt

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


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_dual_encoder(temperature, patience=5, device="cuda"):
    # Device selection (GPU or CPU) with fallback to CPU
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Costruisci il dataset                                                                                                  
    full_dataset = FLAG3DDataset(
    metadata_df=metadata_df,
    annotations_dict=annotations_dict,
    keypoints_data=keypoints_data,
    device=device
)
    # Split dataset in train and val
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)


    # Model
    pose_encoder = TwoStreamAGCN().to(device)
    text_encoder = DistilBERTTextEncoder().to(device)

    # Conteggio parametri
    pose_params = count_parameters(pose_encoder)
    text_params = count_parameters(text_encoder)
    total_params = pose_params + text_params
    print(f"📦 Pose encoder: {pose_params:,} parametri")
    print(f"📚 Text encoder: {text_params:,} parametri")
    print(f"🔢 Totale: {total_params:,} parametri\n")

    # Optimizer
    params = list(pose_encoder.parameters()) + list(text_encoder.parameters())
    optimizer = optim.AdamW(params, lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Logs
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_txt = os.path.join(log_dir, "dual_encoder_training_log.txt")
    log_csv = os.path.join(log_dir, "dual_encoder_training_log.csv")
    
    losses = []
    train_losses = []
    val_losses = []
    
    # Early stopping parameters
    best_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs = 100  # Numero massimo di epoche (early stopping fermerà prima se necessario)

    # Training
    for epoch in range(1, max_epochs + 1):
        pose_encoder.train()
        text_encoder.train()

        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")

        for batch in loop:
            pose_tensor = batch['pose']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            z_pose = pose_encoder(pose_tensor)
            z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute contrastive loss
            loss = ntxent_loss(z_pose, z_text, temperature=temperature)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(train_loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 🔍 VALIDAZIONE
        pose_encoder.eval()
        text_encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pose_tensor = batch['pose']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                z_pose = pose_encoder(pose_tensor)
                z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                loss = ntxent_loss(z_pose, z_text, temperature=temperature)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        losses.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        with open(log_txt, "a") as f:
            f.write(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n")

        
          # ✅ EARLY STOPPING sulla VAL LOSS
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_without_improvement = 0

            torch.save(pose_encoder.state_dict(), os.path.join(log_dir, f"pose_encoder_epoch{epoch}.pt"))
            torch.save(text_encoder.state_dict(), os.path.join(log_dir, f"text_encoder_epoch{epoch}.pt"))

            print(f"\n✅ Miglioramento! Val Loss: {avg_val_loss:.4f} (migliore: {best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"\n⚠️ Nessun miglioramento per {epochs_without_improvement} epoche. Val Loss: {avg_val_loss:.4f} (migliore: {best_loss:.4f})")

        if epochs_without_improvement >= patience:
            print(f"\n🛑 Early stopping attivato dopo {epoch} epoche (nessun miglioramento per {patience} epoche consecutive).")
            break

        # Plot Loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Mostra
        plt.show()

        scheduler.step()

    # Save CSV
    df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses)+1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    df.to_csv(log_csv, index=False)


    # I pesi migliori sono già stati salvati durante il training quando la loss migliorava
    print(f"\n✅ Training completato. Miglior Val Loss: {best_loss:.4f} all'epoca {epoch - epochs_without_improvement}")


if __name__ == "__main__":
    train_dual_encoder(temperature=0.07, patience=5)
