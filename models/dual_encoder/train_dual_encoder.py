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
import numpy as np

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
from utils import GradNormLossWrapper

'''
# Disabilita Flash SDP per forzare il backend a usare l’implementazione classica (compatibile con backward usando GradNorm loss):
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
'''

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
    # Stampa il numero di parametri trainabili
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_grad_stats(model):
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.data.norm(2).item())
    if grad_norms:
        return {
            "mean": np.mean(grad_norms),
            "std": np.std(grad_norms),
            "max": np.max(grad_norms)
        }
    else:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}

def train_dual_encoder(temperature, patience=5, device="cuda"):
    # Device selection (GPU or CPU) with fallback to CPU
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset con augmentation per il training
    train_full_dataset = FLAG3DDataset(
        metadata_df=metadata_df,
        annotations_dict=annotations_dict,
        keypoints_data=keypoints_data,
        device=device
        #augment=False,  # ✅ Abilita augmentation qui
        #augment_prob=0.5
    )

    # Dataset senza augmentation per la validation
    val_full_dataset = FLAG3DDataset(
        metadata_df=metadata_df,
        annotations_dict=annotations_dict,
        keypoints_data=keypoints_data,
        device=device
        #augment=False,  # ❌ Nessuna augmentation
        #augment_prob=0.0
    )

    # Applica split
    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)


    # Model
    pose_encoder = TwoStreamAGCN().to(device)
    text_encoder = DistilBERTTextEncoder(freeze_layers=True, use_adapter=True).to(device)
    
    '''
    # Calcolo delle prime due loss per inizializzare GradNorm
    pose_encoder.eval()
    text_encoder.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        z_pose = pose_encoder(batch['pose'])
        z_text = text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        #initial_loss1 = ntxent_loss(z_pose, z_text, temperature=temperature).item()
        #initial_loss2 = ntxent_loss(z_text, z_pose, temperature=temperature).item()
    '''

    # Conteggio parametri
    pose_params = count_parameters(pose_encoder)
    text_params = count_parameters(text_encoder)
    total_params = pose_params + text_params
    print(f"📦 Pose encoder: {pose_params:,} parametri")
    print(f"📚 Text encoder: {text_params:,} parametri")
    print(f"🔢 Totale: {total_params:,} parametri\n")

    # Parametri da ottimizzare
    params = list(pose_encoder.parameters()) + list(text_encoder.parameters())

    '''
    # Pesi delle due loss (pose2text e text2pose)
    task_weights = torch.nn.Parameter(torch.ones(2, requires_grad=True, device=device))

    # Aggiungere i pesi all’ottimizzatore
    params += [task_weights]
    '''

    # Optimizer
    optimizer = optim.AdamW(params, lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    '''
    # Istanza GradNorm
    initial_losses = [initial_loss1, initial_loss2]
    gradnorm = GradNormLossWrapper(
        loss_fns=[ntxent_loss, ntxent_loss],
        initial_losses=initial_losses,
        alpha=1.5,
        device=device
    )
    '''
    # Logs
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_txt = os.path.join(log_dir, "dual_encoder_training_log.txt")
    log_csv = os.path.join(log_dir, "dual_encoder_training_log.csv")
    
    losses = []
    contrastive_train_losses = []
    train_losses = []
    val_losses = []
    grad_stats = []

    # Early stopping parameters
    best_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs = 100  # Numero massimo di epoche (early stopping fermerà prima se necessario)
    # 🔥 SCHEDULAZIONE TEMPERATURA (decadimento esponenziale)
    tau_start = temperature       # ad esempio 0.2
    gamma = 0.95                  # fattore di decadimento

    # Training
    for epoch in range(1, max_epochs + 1):
        
        # 🔁 Aggiorna temperatura in modo esponenziale
        #current_tau = tau_start * (gamma ** (epoch - 1))
        current_tau = temperature

        pose_encoder.train()
        text_encoder.train()

        train_loss = 0
        contrastive_train_loss = 0
        num_batches = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN] (τ={current_tau:.4f})")

        for batch in loop:
            pose_tensor = batch['pose']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            z_pose = pose_encoder(pose_tensor)
            z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            
            # Due loss separate (pose2text e text2pose)
            loss1 = ntxent_loss(z_pose, z_text, temperature=current_tau)
            #loss2 = ntxent_loss(z_text, z_pose, temperature=current_tau)

            # Calcola la loss bilanciata con GradNorm
            '''
            loss, task_losses, gradnorm_penalty = gradnorm.compute_loss(
                model=None,  
                shared_params=list(pose_encoder.parameters()) + list(text_encoder.parameters()),
                inputs=[z_pose, z_text],
                targets=[z_text, z_pose]
            )'''
            #loss = (loss1 + loss2) / 2



            train_loss += loss1.item() #per loggare la loss
            
            num_batches += 1

            # Aggiorna il postfix con le medie correnti durante il training
            avg_train_loss_current = train_loss / num_batches

            
            loop.set_postfix(
                train_loss=f"{avg_train_loss_current:.4f}")
            
            optimizer.zero_grad()
            loss1.backward()


            pose_grad_stats = compute_grad_stats(pose_encoder)
            text_grad_stats = compute_grad_stats(text_encoder)

            # Logging per epoca
            grad_stats.append({
                "epoch": epoch,
                "pose_grad_mean": pose_grad_stats["mean"],
                "pose_grad_std": pose_grad_stats["std"],
                "pose_grad_max": pose_grad_stats["max"],
                "text_grad_mean": text_grad_stats["mean"],
                "text_grad_std": text_grad_stats["std"],
                "text_grad_max": text_grad_stats["max"],
            })

            optimizer.step()
            
        avg_train_loss = train_loss / len(train_loader)
        #avg_contrastive_train_loss = contrastive_train_loss / len(train_loader)

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
                loss1 = ntxent_loss(z_pose, z_text, temperature=current_tau)
                #loss2 = ntxent_loss(z_text, z_pose, temperature=current_tau)
                #loss = (loss1 + loss2) / 2
                val_loss += loss1.item()

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
            
            # Salva il numero dell'epoca migliore
            with open(os.path.join(log_dir, "best_epoch.txt"), "w") as f:
                f.write(str(epoch))

            print(f"\n✅ Miglioramento! Val Loss: {avg_val_loss:.4f} (migliore: {best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"\n⚠️ Nessun miglioramento per {epochs_without_improvement} epoche. Val Loss: {avg_val_loss:.4f} (migliore: {best_loss:.4f})")

        if epochs_without_improvement >= patience:
            print(f"\n🛑 Early stopping attivato dopo {epoch} epoche (nessun miglioramento per {patience} epoche consecutive).")
            break

       
        scheduler.step()

    # Save CSV
    df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses)+1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    df.to_csv(log_csv, index=False)

    # Save grad stats
    df_grad_stats = pd.DataFrame(grad_stats)
    df_grad_stats.to_csv(os.path.join(log_dir, "grad_stats.csv"), index=False)

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

    # Salva e mostra il grafico
    plt.savefig(os.path.join(log_dir, f"train_val_loss_plot_epoch{epoch}.png"))
    plt.show()

    # I pesi migliori sono già stati salvati durante il training quando la loss migliorava
    print(f"\n✅ Training completato. Miglior Val Loss: {best_loss:.4f} all'epoca {epoch - epochs_without_improvement}")


if __name__ == "__main__":
    train_dual_encoder(temperature=0.07, patience=5)
