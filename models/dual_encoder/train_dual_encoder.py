# train_dual_encoder.py

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



def train_dual_encoder(temperature, device="cuda"):
    # Device selection (GPU or CPU) with fallback to CPU
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Costruisci il dataset
    dataset = FLAG3DDataset(
    metadata_df=metadata_df,
    annotations_dict=annotations_dict,
    keypoints_data=keypoints_data,
    device=device
)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Model
    pose_encoder = TwoStreamAGCN().to(device)
    text_encoder = DistilBERTTextEncoder().to(device)

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
    
    # Early stopping parameters
    best_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    max_epochs = 100  # Numero massimo di epoche (early stopping fermerà prima se necessario)

    # Training
    for epoch in range(1, max_epochs + 1):
        pose_encoder.train()
        text_encoder.train()

        epoch_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in loop:
            pose_tensor = batch['pose']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            z_pose = pose_encoder(pose_tensor)
            z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute contrastive loss
            loss = ntxent_loss(z_pose, z_text, temperature=temperature)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        losses.append({"epoch": epoch, "loss": avg_loss})

        with open(log_txt, "a") as f:
            f.write(f"Epoch {epoch}, Loss: {avg_loss:.4f}\n")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            # Salva i migliori pesi quando la loss migliora
            torch.save(pose_encoder.state_dict(), os.path.join(log_dir, "pose_encoder.pt"))
            torch.save(text_encoder.state_dict(), os.path.join(log_dir, "text_encoder.pt"))
            print(f"\n✅ Miglioramento! Loss: {avg_loss:.4f} (migliore: {best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"\n⚠️  Nessun miglioramento per {epochs_without_improvement} epoche. Loss: {avg_loss:.4f} (migliore: {best_loss:.4f})")
        
        # Early stopping: ferma se non c'è miglioramento per 'patience' epoche
        if epochs_without_improvement >= patience:
            print(f"\n🛑 Early stopping attivato dopo {epoch} epoche (nessun miglioramento per {patience} epoche consecutive).")
            break

    # Save CSV
    df = pd.DataFrame(losses)
    df.to_csv(log_csv, index=False)

    # Validation su un batch
    pose_encoder.eval()
    text_encoder.eval()

    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        pose_tensor = sample_batch['pose'].to(device)
        input_ids = sample_batch['input_ids'].to(device)
        attention_mask = sample_batch['attention_mask'].to(device)

        z_pose = pose_encoder(pose_tensor)
        z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        val_loss = ntxent_loss(z_pose, z_text, temperature=temperature).item()

        print(f"\n🔍 Loss di validazione su batch casuale: {val_loss:.4f}")

    # I pesi migliori sono già stati salvati durante il training quando la loss migliorava
    print(f"\n✅ Training completato. Miglior loss: {best_loss:.4f} all'epoca {epoch - epochs_without_improvement}")


if __name__ == "__main__":
    train_dual_encoder(temperature=0.07)
