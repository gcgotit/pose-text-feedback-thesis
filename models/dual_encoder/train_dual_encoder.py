"""
train_dual_encoder.py
---------------------
Addestramento del dual encoder (2S-AGCN + DistilBERT) su FLAG3D.
Include:
- Train/val split con Subset
- Loss contrastiva NT-Xent
- Early stopping su validation loss
- Salvataggio log, pesi, grafici
- 🆕 Mixed Precision (AMP) per ridurre memoria GPU
- 🆕 torch.compile per velocizzare il training (PyTorch 2.0+)
- 🆕 Freezing di DistilBERT con Adapter trainabile
"""

from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
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
from models.pose_encoder.twostream_stgcn_plus import TwoStreamSTGCNPlusEncoder
from models.text_encoder.distilbert_adapter import DistilBERTTextEncoder
from data.FLAG3D.flag3d_dataset import FLAG3DDataset
from utils import ntxent_loss  # contrastive loss
import json
import pickle
import matplotlib.pyplot as plt
from data.FLAG3D.load_augmented_data import load_flag3d_data


# Percorso alla directory dei dati
data_dir = project_root / "data" / "FLAG3D"


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
        return sum(p.numel() for p in model.parameters())

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


def train_dual_encoder(
    temperature, 
    patience=8, 
    device="cuda", 
    use_augmented=True, 
    encoder_type="stgcn_plus",
    freeze_bert=True,
    use_adapter=True,
    use_amp=True,
    use_compile=True,
    use_resampling=False,
    target_frames=300,
    resample_method="linear"
):
    """
    Addestra il dual encoder (pose + text) con loss contrastiva NT-Xent.
    
    Args:
        temperature: Temperatura per la loss NT-Xent
        patience: Numero di epoche senza miglioramento prima dell'early stopping
        device: Device per il training ("cuda" o "cpu")
        use_augmented: Se True, usa i dati aumentati offline (triplicando il training set)
        encoder_type: Tipo di pose encoder da usare:
            - "agcn": TwoStreamAGCN (encoder originale, ~50K parametri)
            - "stgcn_plus": TwoStreamSTGCNPlusEncoder (encoder potenziato, ~1.5M parametri)
        freeze_bert: Se True, congela DistilBERT (solo adapter e projection trainabili)
        use_adapter: Se True, aggiunge un Adapter MLP trainabile al text encoder
        use_amp: Se True, usa Mixed Precision (AMP) per ridurre memoria GPU
        use_compile: Se True, usa torch.compile per velocizzare (richiede PyTorch 2.0+)
    """
    # Device selection (GPU or CPU) with fallback to CPU
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Verifica versione PyTorch per torch.compile
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    can_compile = torch_version >= (2, 0)
    if use_compile and not can_compile:
        print(f"⚠️ torch.compile richiede PyTorch 2.0+, trovato {torch.__version__}. Disabilitato.")
        use_compile = False

    # Carica i dati (originali o combinati con augmentation)
    if use_resampling:
        print(f"\n📥 Caricamento dati CON RESAMPLING a T={target_frames}...")
        from data.FLAG3D.load_augmented_data import load_flag3d_data_resampled
        keypoints_data, metadata_df, annotations_dict, split = load_flag3d_data_resampled(
            target_frames=target_frames,
            data_dir=data_dir
        )
    else:
        print(f"\n📥 Caricamento dati {'COMBINATI (originali + aumentati)' if use_augmented else 'ORIGINALI'}...")
        keypoints_data, metadata_df, annotations_dict, split = load_flag3d_data(
            use_augmented=use_augmented,
            data_dir=data_dir
        )
    
    train_indices = split["train_indices"]
    val_indices = split["val_indices"]
    
    print(f"   - Training samples: {len(train_indices)}")
    print(f"   - Validation samples: {len(val_indices)}")
    if use_resampling:
        print(f"   - Resampling: T={target_frames}, method={resample_method}")
    if use_augmented and "augmented_train_count" in split:
        print(f"   - (di cui {split['original_train_count']} originali + {split['augmented_train_count']} aumentati)")

    # Costruisci il dataset                                                                                                  
    full_dataset = FLAG3DDataset(
        metadata_df=metadata_df,
        annotations_dict=annotations_dict,
        keypoints_data=keypoints_data,
        device=device,
        use_resampling=use_resampling,
        target_frames=target_frames,
        resample_method=resample_method
    )
    # Split dataset in train and val
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)


    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL SETUP
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Pose Encoder
    if encoder_type == "agcn":
        print("🧠 Usando TwoStreamAGCN (encoder originale)")
        pose_encoder = TwoStreamAGCN().to(device)
    elif encoder_type == "stgcn_plus":
        print("🧠 Usando TwoStreamSTGCNPlusEncoder (encoder potenziato)")
        pose_encoder = TwoStreamSTGCNPlusEncoder(
            input_dim=3,
            hidden_channels=[64, 128, 256, 256],
            output_dim=128,
            num_nodes=25,
            dropout=0.1,
            fusion_dropout=0.3
        ).to(device)
    else:
        raise ValueError(f"encoder_type non valido: {encoder_type}. Usa 'agcn' o 'stgcn_plus'.")
    
    # Text Encoder (con freezing e adapter opzionali)
    print(f"\n{'🧊' if freeze_bert else '🔥'} Text Encoder: freeze_bert={freeze_bert}, use_adapter={use_adapter}")
    text_encoder = DistilBERTTextEncoder(
        freeze_bert=freeze_bert,
        use_adapter=use_adapter,
        adapter_bottleneck=256
    ).to(device)

    # ═══════════════════════════════════════════════════════════════════════════
    # TORCH.COMPILE (PyTorch 2.0+)
    # ═══════════════════════════════════════════════════════════════════════════
    if use_compile:
        print("\n⚡ Compilazione modelli con torch.compile...")
        try:
            pose_encoder = torch.compile(pose_encoder)
            text_encoder = torch.compile(text_encoder)
            print("   ✓ Modelli compilati con successo")
        except Exception as e:
            print(f"   ⚠️ Errore durante la compilazione: {e}")
            print("   Continuando senza torch.compile...")

    # Conteggio parametri
    pose_params = count_parameters(pose_encoder)
    text_params = count_parameters(text_encoder)
    text_total = count_total_parameters(text_encoder)
    total_params = pose_params + text_params
    
    print(f"\n📦 Pose encoder: {pose_params:,} parametri trainabili")
    print(f"📚 Text encoder: {text_params:,} trainabili / {text_total:,} totali")
    print(f"🔢 Totale trainabili: {total_params:,} parametri\n")

    # Optimizer (solo parametri trainabili)
    params = [p for p in pose_encoder.parameters() if p.requires_grad] + \
             [p for p in text_encoder.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # ═══════════════════════════════════════════════════════════════════════════
    # MIXED PRECISION SETUP
    # ═══════════════════════════════════════════════════════════════════════════
    scaler = GradScaler('cuda') if use_amp and device.type == "cuda" else None
    if use_amp:
        if device.type == "cuda":
            print("⚡ Mixed Precision (AMP) attivato")
        else:
            print("⚠️ AMP richiede CUDA, disabilitato su CPU")
            use_amp = False

    # Logs
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_txt = os.path.join(log_dir, "dual_encoder_training_log.txt")
    log_csv = os.path.join(log_dir, "dual_encoder_training_log.csv")
    
    losses = []
    train_losses = []
    val_losses = []
    grad_stats = []

    # Early stopping parameters
    best_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs = 100  # Numero massimo di epoche (early stopping fermerà prima se necessario)

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    for epoch in range(1, max_epochs + 1):
        pose_encoder.train()
        text_encoder.train()

        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")

        for batch in loop:
            pose_tensor = batch['pose']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            optimizer.zero_grad()
            
            # Forward pass con Mixed Precision (se abilitato)
            if use_amp and scaler is not None:
                with autocast(device_type='cuda'):
                    z_pose = pose_encoder(pose_tensor)
                    z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    loss = ntxent_loss(z_pose, z_text, temperature=temperature)
                
                # Backward pass con scaler
                scaler.scale(loss).backward()
                
                # Gradient stats (prima di unscale per evitare inf)
                scaler.unscale_(optimizer)
                pose_grad_stats = compute_grad_stats(pose_encoder)
                text_grad_stats = compute_grad_stats(text_encoder)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward/backward senza AMP
                z_pose = pose_encoder(pose_tensor)
                z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                loss = ntxent_loss(z_pose, z_text, temperature=temperature)
                
                loss.backward()
                
                pose_grad_stats = compute_grad_stats(pose_encoder)
                text_grad_stats = compute_grad_stats(text_encoder)
                
                optimizer.step()
            
            train_loss += loss.item()

            # Logging per batch
            grad_stats.append({
                "epoch": epoch,
                "pose_grad_mean": pose_grad_stats["mean"],
                "pose_grad_std": pose_grad_stats["std"],
                "pose_grad_max": pose_grad_stats["max"],
                "text_grad_mean": text_grad_stats["mean"],
                "text_grad_std": text_grad_stats["std"],
                "text_grad_max": text_grad_stats["max"],
            })
            
            loop.set_postfix(train_loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # ═══════════════════════════════════════════════════════════════════════
        # VALIDATION
        # ═══════════════════════════════════════════════════════════════════════
        pose_encoder.eval()
        text_encoder.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pose_tensor = batch['pose']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                # Validation con AMP (solo autocast, no scaler)
                if use_amp and device.type == "cuda":
                    with autocast(device_type='cuda'):
                        z_pose = pose_encoder(pose_tensor)
                        z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                        loss = ntxent_loss(z_pose, z_text, temperature=temperature)
                else:
                    z_pose = pose_encoder(pose_tensor)
                    z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    loss = ntxent_loss(z_pose, z_text, temperature=temperature)
                
                val_loss += loss.item()

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

        
        # ═══════════════════════════════════════════════════════════════════════
        # EARLY STOPPING sulla VAL LOSS
        # ═══════════════════════════════════════════════════════════════════════
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            
            # Salva i pesi (gestisce sia modelli compilati che non)
            pose_state = pose_encoder._orig_mod.state_dict() if hasattr(pose_encoder, '_orig_mod') else pose_encoder.state_dict()
            text_state = text_encoder._orig_mod.state_dict() if hasattr(text_encoder, '_orig_mod') else text_encoder.state_dict()
            
            torch.save(pose_state, os.path.join(log_dir, f"pose_encoder_epoch{epoch}.pt"))
            torch.save(text_state, os.path.join(log_dir, f"text_encoder_epoch{epoch}.pt"))
            
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

    train_dual_encoder(
        temperature=0.10,
        patience=8,
        device="cuda",
        use_augmented=True,
        encoder_type="stgcn_plus",
        # 🆕 Ottimizzazioni memoria/velocità
        freeze_bert=True,      # Congela DistilBERT (riduce memoria e parametri trainabili)
        use_adapter=True,      # Adapter MLP per contrastive learning
        use_amp=True,          # Mixed Precision (riduce memoria ~50%)
        use_compile=True,       # torch.compile (velocizza ~20-30% su PyTorch 2.0+)

        # 🆕 Attiva resampling uniforme a 300 frame
        use_resampling=True,
        target_frames=300,
        resample_method="linear"
    )
