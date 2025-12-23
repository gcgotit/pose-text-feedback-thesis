"""
eval_embedding_analysis.py
---------------------------
Analisi approfondita del comportamento del dual encoder sul test/validation set.

Include:
1. Analisi qualitativa dei Retrieval Errors (Recall@1 failures)
2. Visualizzazione t-SNE/PCA con match/errori evidenziati
3. Distribuzione delle similarità (match vs non-match)
4. Audit delle sequenze di pose (varianza, outlier)
5. Linear Probe Classification (Top-1, Top-5, F1-macro, AUC, Confusion Matrix)
6. Retrieval bidirezionale (Text→Pose e Pose→Text con Mean/Median Rank)
7. Analisi per classe (metriche per-class, confusion matrix retrieval)

Riferimenti:
- Zhao et al. (ACCV 2022): 90.9% accuracy su identificazione errori esercizio
- Metriche standard retrieval cross-modale: Recall@K, mAP, Mean/Median Rank

Uso:
    python eval_embedding_analysis.py --use_augmented
    python eval_embedding_analysis.py --max_samples 500 --tsne_samples 200
    
I pesi del modello vengono caricati automaticamente dalla best epoch (logs/best_epoch.txt).
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Seaborn è opzionale per i KDE plot
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️ seaborn non installato. I KDE plot useranno matplotlib. Installa con: pip install seaborn")
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    pairwise_distances, 
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, normalize
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Aggiunge la root del progetto al PYTHONPATH
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.FLAG3D.flag3d_dataset import FLAG3DDataset
from data.FLAG3D.load_augmented_data import load_flag3d_data
from models.pose_encoder.twostream_stgcn_plus import TwoStreamSTGCNPlusEncoder
from models.text_encoder.distilbert_adapter import DistilBERTTextEncoder


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE E SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# FLAG3D skeleton connections (25 joints)
# Based on standard skeleton topology
SKELETON_CONNECTIONS = [
    # Torso
    (0, 1), (1, 20), (20, 2), (2, 3),  # spine
    # Left arm
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    # Right arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    # Left leg
    (0, 12), (12, 13), (13, 14), (14, 15),
    # Right leg
    (0, 16), (16, 17), (17, 18), (18, 19),
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analisi embedding dual encoder')
    
    parser.add_argument('--use_augmented', action='store_true',
                        help='Usa dati combinati (originali + aumentati)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per embedding extraction')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Numero massimo di sample da analizzare (None = tutti)')
    parser.add_argument('--tsne_samples', type=int, default=500,
                        help='Numero di sample per t-SNE (subset bilanciato)')
    parser.add_argument('--top_k_errors', type=int, default=20,
                        help='Numero massimo di errori da visualizzare')
    
    return parser.parse_args()


def setup_directories():
    """Crea le directory di output."""
    dirs = [
        project_root / "debug_plots" / "qualitative_errors",
        project_root / "debug_outputs"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs[0].parent, dirs[1]


def load_label_names() -> Dict[str, str]:
    """Carica la mappatura label -> nome azione."""
    label_path = project_root / "data" / "FLAG3D" / "flag3d_label_names.json"
    with open(label_path, 'r') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNZIONI DI ESTRAZIONE EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

def load_models(device: torch.device):
    """
    Carica i modelli pose encoder e text encoder con i pesi della best epoch.
    Usa lo stesso setup del training (stgcn_plus + DistilBERT freezato con adapter).
    """
    # Inizializza i modelli con la stessa configurazione del training
    pose_encoder = TwoStreamSTGCNPlusEncoder(
        input_dim=3,
        hidden_channels=[64, 128, 256, 256],
        output_dim=128,
        num_nodes=25,
        dropout=0.1,
        fusion_dropout=0.3
    ).to(device)
    
    text_encoder = DistilBERTTextEncoder(
        freeze_bert=True,
        use_adapter=True,
        adapter_bottleneck=256
    ).to(device)
    
    log_dir = project_root / "logs"
    
    # Carica il numero dell'epoca migliore
    with open(log_dir / "best_epoch.txt") as f:
        best_epoch = int(f.read().strip())
    
    # Carica i pesi migliori
    pose_encoder.load_state_dict(torch.load(log_dir / f"pose_encoder_epoch{best_epoch}.pt", map_location=device))
    text_encoder.load_state_dict(torch.load(log_dir / f"text_encoder_epoch{best_epoch}.pt", map_location=device))
    
    print(f"📂 Caricati pesi della best epoch: {best_epoch}")
    
    # Setta i modelli in eval mode
    pose_encoder.eval()
    text_encoder.eval()
    
    return pose_encoder, text_encoder


def extract_embeddings(
    pose_encoder: torch.nn.Module,
    text_encoder: torch.nn.Module,
    dataset: Subset,
    device: torch.device,
    batch_size: int = 32,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], List[int]]:
    """
    Estrae le embedding dal dataset.
    
    Returns:
        pose_embeds: (N, 128) embedding delle pose
        text_embeds: (N, 128) embedding dei testi
        labels: (N,) etichette delle classi
        metadata: lista di dizionari con action_id, text, ecc.
        indices: indici nel dataset originale
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_pose_embeds = []
    all_text_embeds = []
    all_labels = []
    all_metadata = []
    all_indices = []
    
    total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Estrazione embedding")):
            if max_samples and processed >= max_samples:
                break
            
            pose_tensor = batch['pose'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            z_pose = pose_encoder(pose_tensor)
            z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            
            all_pose_embeds.append(z_pose.cpu().numpy())
            all_text_embeds.append(z_text.cpu().numpy())
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            
            # Calcola gli indici globali
            start_idx = batch_idx * batch_size
            batch_indices = list(range(start_idx, start_idx + len(labels)))
            all_indices.extend(batch_indices)
            
            processed += len(labels)
    
    pose_embeds = np.vstack(all_pose_embeds)
    text_embeds = np.vstack(all_text_embeds)
    labels = np.array(all_labels)
    
    return pose_embeds, text_embeds, labels, all_metadata, all_indices


def compute_retrieval_results(
    pose_embeds: np.ndarray,
    text_embeds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola i risultati di retrieval text->pose.
    
    Returns:
        sim_matrix: (N, N) matrice delle similarità coseno
        ranks: (N,) rank della pose corretta per ogni query
        top_k_indices: (N, k) indici delle top-k pose per ogni query
    """
    # Normalizza per similarità coseno
    pose_norm = normalize(pose_embeds)
    text_norm = normalize(text_embeds)
    
    # Matrice di similarità (text_i, pose_j)
    sim_matrix = text_norm @ pose_norm.T
    
    # Per ogni query (testo), trova il rank della pose corretta
    ranks = []
    k = 10
    top_k_indices = []
    
    for i in range(len(text_embeds)):
        sorted_idx = np.argsort(-sim_matrix[i])  # Ordine decrescente
        rank = np.where(sorted_idx == i)[0][0] + 1  # 1-based rank
        ranks.append(rank)
        top_k_indices.append(sorted_idx[:k])
    
    return sim_matrix, np.array(ranks), np.array(top_k_indices)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ANALISI QUALITATIVA DEI RETRIEVAL ERRORS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_skeleton_2d(ax, keypoints: np.ndarray, title: str, color: str = 'blue', alpha: float = 1.0):
    """
    Disegna uno skeleton 2D (usa x, y dalle coordinate 3D).
    
    Args:
        ax: matplotlib axis
        keypoints: (T, 25, 3) sequenza di keypoint
        title: titolo del subplot
        color: colore dei joint e bones
        alpha: trasparenza
    """
    # Usa il frame centrale per la visualizzazione
    # Trova un frame non-padding (varianza > 0)
    valid_frames = np.where(np.var(keypoints, axis=(1, 2)) > 1e-6)[0]
    if len(valid_frames) == 0:
        ax.text(0.5, 0.5, 'No valid frames', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    frame_idx = valid_frames[len(valid_frames) // 2]  # Frame centrale
    kp = keypoints[frame_idx]  # (25, 3)
    
    # Estrai x, y (ignora z per la visualizzazione 2D)
    x = kp[:, 0]
    y = kp[:, 1]
    
    # Disegna le connessioni
    for (j1, j2) in SKELETON_CONNECTIONS:
        if j1 < len(x) and j2 < len(x):
            ax.plot([x[j1], x[j2]], [y[j1], y[j2]], c=color, linewidth=2, alpha=alpha)
    
    # Disegna i joint
    ax.scatter(x, y, c=color, s=30, zorder=5, alpha=alpha)
    
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Y cresce verso il basso per la visualizzazione pose
    ax.axis('off')


def analyze_retrieval_errors(
    pose_embeds: np.ndarray,
    text_embeds: np.ndarray,
    labels: np.ndarray,
    dataset: Subset,
    label_names: Dict[str, str],
    annotations_dict: Dict,
    sim_matrix: np.ndarray,
    ranks: np.ndarray,
    top_k_indices: np.ndarray,
    output_dir: Path,
    max_errors: int = 20
):
    """
    Analizza e visualizza i Recall@1 failures.
    """
    print("\n" + "=" * 70)
    print("📊 ANALISI QUALITATIVA RETRIEVAL ERRORS (Recall@1 Failures)")
    print("=" * 70)
    
    # Trova gli errori (rank > 1)
    error_indices = np.where(ranks > 1)[0]
    n_errors = len(error_indices)
    n_total = len(ranks)
    
    print(f"\n📈 Statistiche:")
    print(f"   - Totale sample: {n_total}")
    print(f"   - Recall@1 failures: {n_errors} ({100 * n_errors / n_total:.1f}%)")
    print(f"   - Recall@1: {100 * (1 - n_errors / n_total):.2f}%")
    
    errors_dir = output_dir / "qualitative_errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    
    # Ordina per rank (peggiori prima)
    sorted_errors = error_indices[np.argsort(-ranks[error_indices])]
    
    error_logs = []
    
    for idx, err_idx in enumerate(sorted_errors[:max_errors]):
        sample = dataset[err_idx]
        label = labels[err_idx]
        action_name = label_names.get(str(label), f"Unknown_{label}")
        
        # Estrai action_id dal dataset sottostante
        full_dataset = dataset.dataset
        original_idx = dataset.indices[err_idx]
        row = full_dataset.df.iloc[original_idx]
        action_id = row['action_id']
        
        # Ottieni la descrizione testuale
        text_info = annotations_dict.get(str(action_id), {})
        text_description = text_info.get('name', 'N/A') + ": " + text_info.get('description', 'N/A')[:100]
        
        # Similarità con ground truth
        gt_sim = sim_matrix[err_idx, err_idx]
        
        # Top-5 retrieved
        top5_idx = top_k_indices[err_idx][:5]
        top5_sims = sim_matrix[err_idx, top5_idx]
        top5_labels = labels[top5_idx]
        
        # Log
        log_entry = {
            'sample_idx': int(err_idx),
            'action_id': int(action_id),
            'label': int(label),
            'action_name': action_name,
            'rank': int(ranks[err_idx]),
            'gt_similarity': float(gt_sim),
            'text_preview': text_description[:100],
            'top5_labels': [int(l) for l in top5_labels],
            'top5_sims': [float(s) for s in top5_sims]
        }
        error_logs.append(log_entry)
        
        # Print dettagli
        print(f"\n{'─' * 60}")
        print(f"🔴 Error #{idx + 1} | Sample {err_idx}")
        print(f"   Action ID: {action_id} | Label: {label} ({action_name})")
        print(f"   GT Cosine Similarity: {gt_sim:.4f} | Rank: {ranks[err_idx]}")
        print(f"   Text: {text_description}")
        print(f"\n   Top-5 Retrieved:")
        for j, (ret_idx, ret_sim) in enumerate(zip(top5_idx, top5_sims)):
            ret_label = labels[ret_idx]
            ret_name = label_names.get(str(ret_label), f"Unknown_{ret_label}")
            match_str = "✓ MATCH" if ret_label == label else "✗"
            print(f"      {j + 1}. Pose {ret_idx} | Label {ret_label} ({ret_name}) | Sim: {ret_sim:.4f} {match_str}")
        
        # Plot skeleton
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'Error #{idx + 1}: {action_name} (Rank={ranks[err_idx]})', fontsize=12)
        
        # Ground truth pose
        gt_pose = sample['pose'].cpu().numpy()  # (T, 25, 3)
        gt_pose = gt_pose.transpose(1, 2, 0).transpose(2, 0, 1)  # riporta a (T, 25, 3)
        plot_skeleton_2d(axes[0, 0], gt_pose, 'Ground Truth Pose', color='green')
        
        # Top-5 retrieved poses
        for j, (ret_idx, ret_sim) in enumerate(zip(top5_idx[:5], top5_sims[:5])):
            row_idx = j // 3
            if j < 3:
                row_idx = 0
                col_idx = j  # Ma abbiamo già usato 0,0 per GT
            ax_idx = j + 1 if j < 2 else j
            ax = axes.flat[j + 1]
            
            ret_sample = dataset[ret_idx]
            ret_pose = ret_sample['pose'].cpu().numpy()
            ret_pose = ret_pose.transpose(1, 2, 0).transpose(2, 0, 1)
            
            ret_label = labels[ret_idx]
            ret_name = label_names.get(str(ret_label), f"?")
            color = 'blue' if ret_label == label else 'red'
            plot_skeleton_2d(ax, ret_pose, f'#{j + 1}: {ret_name[:20]}... (sim={ret_sim:.3f})', color=color)
        
        plt.tight_layout()
        plt.savefig(errors_dir / f"error_{idx + 1:03d}_sample{err_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Salva log degli errori
    errors_df = pd.DataFrame(error_logs)
    errors_df.to_csv(errors_dir.parent.parent / "debug_outputs" / "retrieval_errors_log.csv", index=False)
    
    print(f"\n✅ Salvate {min(max_errors, n_errors)} visualizzazioni in {errors_dir}")
    return error_logs


# ═══════════════════════════════════════════════════════════════════════════════
# 2. T-SNE / PCA CON MATCH E ERRORI EVIDENZIATI
# ═══════════════════════════════════════════════════════════════════════════════

def create_balanced_subset(
    pose_embeds: np.ndarray,
    text_embeds: np.ndarray,
    labels: np.ndarray,
    ranks: np.ndarray,
    n_samples: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea un subset bilanciato per t-SNE.
    """
    unique_labels = np.unique(labels)
    samples_per_class = max(1, n_samples // len(unique_labels))
    
    selected_indices = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        n_select = min(samples_per_class, len(label_indices))
        selected = np.random.choice(label_indices, n_select, replace=False)
        selected_indices.extend(selected)
    
    selected_indices = np.array(selected_indices)[:n_samples]
    
    return (
        pose_embeds[selected_indices],
        text_embeds[selected_indices],
        labels[selected_indices],
        ranks[selected_indices]
    )


def plot_tsne_with_matches(
    pose_embeds: np.ndarray,
    text_embeds: np.ndarray,
    labels: np.ndarray,
    ranks: np.ndarray,
    label_names: Dict[str, str],
    output_path: Path,
    method: str = 'tsne'
):
    """
    Visualizza t-SNE/PCA con match corretti e errori evidenziati.
    """
    print(f"\n{'=' * 70}")
    print(f"📐 VISUALIZZAZIONE {method.upper()}")
    print("=" * 70)
    
    # Combina pose e text embedding
    n_samples = len(pose_embeds)
    X_combined = np.vstack([pose_embeds, text_embeds])
    X_normalized = normalize(X_combined)
    
    # Dimensionality reduction
    print(f"   Applicando {method.upper()} su {len(X_combined)} punti...")
    if method == 'tsne':
        # Perplexity deve essere < n_samples
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, 
                       random_state=42, init='pca', max_iter=1000)
    else:
        reducer = PCA(n_components=2)
    
    X_2d = reducer.fit_transform(X_normalized)
    
    # Split in pose e text
    pose_2d = X_2d[:n_samples]
    text_2d = X_2d[n_samples:]
    
    # Identifica match corretti e errori
    correct_matches = ranks == 1
    errors = ranks > 1
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colormap per le classi
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    cmap_name = 'tab20' if n_classes <= 20 else 'hsv'
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = {label: cmap(i / n_classes) for i, label in enumerate(unique_labels)}
    
    # Disegna linee per i match
    print("   Disegnando connessioni pose-text...")
    for i in range(n_samples):
        color = 'lightgray' if correct_matches[i] else 'red'
        linestyle = '-' if correct_matches[i] else '--'
        alpha = 0.3 if correct_matches[i] else 0.7
        linewidth = 0.5 if correct_matches[i] else 1.5
        
        ax.plot([pose_2d[i, 0], text_2d[i, 0]], 
                [pose_2d[i, 1], text_2d[i, 1]],
                c=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth, zorder=1)
    
    # Plot pose (cerchi) e text (triangoli) per classe
    for label in unique_labels:
        mask = labels == label
        label_name = label_names.get(str(label), f"Class {label}")[:15]
        
        # Pose (cerchi)
        ax.scatter(pose_2d[mask, 0], pose_2d[mask, 1],
                   c=[colors[label]], marker='o', s=50, alpha=0.7,
                   edgecolors='white', linewidths=0.5, zorder=2)
        
        # Text (triangoli)
        ax.scatter(text_2d[mask, 0], text_2d[mask, 1],
                   c=[colors[label]], marker='^', s=50, alpha=0.7,
                   edgecolors='white', linewidths=0.5, zorder=2)
    
    # Evidenzia errori
    error_mask = errors
    ax.scatter(pose_2d[error_mask, 0], pose_2d[error_mask, 1],
               c='none', marker='o', s=150, edgecolors='red', linewidths=2, zorder=3)
    ax.scatter(text_2d[error_mask, 0], text_2d[error_mask, 1],
               c='none', marker='^', s=150, edgecolors='red', linewidths=2, zorder=3)
    
    # Legenda custom
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Pose'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=10, label='Text'),
        Line2D([0], [0], color='lightgray', linewidth=2, label='Match corretto'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Match errato'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='red',
               markerfacecolor='none', markersize=12, markeredgewidth=2, label='Errore')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Titolo e labels
    ax.set_title(f'{method.upper()} Embedding Space - Match vs Errors\n'
                 f'({sum(correct_matches)} correct, {sum(errors)} errors out of {n_samples} samples)',
                 fontsize=14)
    ax.set_xlabel(f'{method.upper()} Dimension 1')
    ax.set_ylabel(f'{method.upper()} Dimension 2')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Salvato {method.upper()} plot in {output_path}")
    
    if method == 'pca' and hasattr(reducer, 'explained_variance_ratio_'):
        print(f"   Varianza spiegata: {reducer.explained_variance_ratio_}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUZIONE DELLE SIMILARITÀ
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_similarity_distribution(
    sim_matrix: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    k_distractors: int = 5
):
    """
    Analizza e visualizza la distribuzione delle similarità match vs non-match.
    """
    print("\n" + "=" * 70)
    print("📊 DISTRIBUZIONE SIMILARITÀ (Match vs Non-Match)")
    print("=" * 70)
    
    n = len(labels)
    
    # Similarità con ground truth (diagonale)
    match_sims = np.diag(sim_matrix)
    
    # Similarità con i distrattori (non sulla diagonale)
    non_match_sims = []
    top_k_distractor_sims = []
    
    for i in range(n):
        # Tutti i non-match per questo sample
        row_sims = sim_matrix[i].copy()
        row_sims[i] = -np.inf  # Escludi il match
        non_match_sims.extend(row_sims[row_sims > -np.inf])
        
        # Top-k distrattori
        sorted_idx = np.argsort(-row_sims)
        top_k_distractor_sims.append(row_sims[sorted_idx[:k_distractors]])
    
    non_match_sims = np.array(non_match_sims)
    top_k_distractor_sims = np.array(top_k_distractor_sims).flatten()
    
    # Statistiche
    print(f"\n📈 Statistiche Similarità:")
    print(f"   Match (ground truth):")
    print(f"      - Mean: {np.mean(match_sims):.4f}")
    print(f"      - Std:  {np.std(match_sims):.4f}")
    print(f"      - Min:  {np.min(match_sims):.4f}")
    print(f"      - Max:  {np.max(match_sims):.4f}")
    
    print(f"\n   Non-Match (all distractors):")
    print(f"      - Mean: {np.mean(non_match_sims):.4f}")
    print(f"      - Std:  {np.std(non_match_sims):.4f}")
    print(f"      - Min:  {np.min(non_match_sims):.4f}")
    print(f"      - Max:  {np.max(non_match_sims):.4f}")
    
    print(f"\n   Top-{k_distractors} Distractors:")
    print(f"      - Mean: {np.mean(top_k_distractor_sims):.4f}")
    print(f"      - Std:  {np.std(top_k_distractor_sims):.4f}")
    
    # Separabilità
    margin = np.mean(match_sims) - np.mean(top_k_distractor_sims)
    print(f"\n   📏 Margin (mean_match - mean_top{k_distractors}_distractor): {margin:.4f}")
    
    # Calcola AUC per la separabilità
    # 1 = match, 0 = non-match
    y_true = np.concatenate([np.ones(len(match_sims)), np.zeros(len(top_k_distractor_sims))])
    y_score = np.concatenate([match_sims, top_k_distractor_sims])
    auc = roc_auc_score(y_true, y_score)
    print(f"   🎯 AUC (Match vs Top-{k_distractors} Distractors): {auc:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    bins = np.linspace(-0.5, 1.0, 50)
    ax1.hist(match_sims, bins=bins, alpha=0.7, label=f'Match (GT)', color='green', density=True)
    ax1.hist(top_k_distractor_sims, bins=bins, alpha=0.7, 
             label=f'Top-{k_distractors} Distractors', color='red', density=True)
    ax1.axvline(np.mean(match_sims), color='darkgreen', linestyle='--', linewidth=2,
                label=f'Mean Match: {np.mean(match_sims):.3f}')
    ax1.axvline(np.mean(top_k_distractor_sims), color='darkred', linestyle='--', linewidth=2,
                label=f'Mean Distractor: {np.mean(top_k_distractor_sims):.3f}')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution: Match vs Top-K Distractors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # KDE Plot
    ax2 = axes[1]
    if HAS_SEABORN:
        try:
            sns.kdeplot(match_sims, ax=ax2, label='Match (GT)', color='green', fill=True, alpha=0.3)
            sns.kdeplot(top_k_distractor_sims, ax=ax2, label=f'Top-{k_distractors} Distractors', 
                        color='red', fill=True, alpha=0.3)
        except Exception:
            # Fallback to histogram if KDE fails
            ax2.hist(match_sims, bins=30, alpha=0.5, label='Match', color='green', density=True)
            ax2.hist(top_k_distractor_sims, bins=30, alpha=0.5, label='Distractor', color='red', density=True)
    else:
        # Fallback senza seaborn: usa scipy per KDE manuale
        try:
            from scipy.stats import gaussian_kde
            x_range = np.linspace(-0.5, 1.0, 200)
            
            kde_match = gaussian_kde(match_sims)
            kde_distractor = gaussian_kde(top_k_distractor_sims)
            
            ax2.fill_between(x_range, kde_match(x_range), alpha=0.3, color='green', label='Match (GT)')
            ax2.fill_between(x_range, kde_distractor(x_range), alpha=0.3, color='red', label=f'Top-{k_distractors} Distractors')
            ax2.plot(x_range, kde_match(x_range), color='green', linewidth=2)
            ax2.plot(x_range, kde_distractor(x_range), color='red', linewidth=2)
        except Exception:
            ax2.hist(match_sims, bins=30, alpha=0.5, label='Match', color='green', density=True)
            ax2.hist(top_k_distractor_sims, bins=30, alpha=0.5, label='Distractor', color='red', density=True)
    
    ax2.axvline(np.mean(match_sims), color='darkgreen', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(top_k_distractor_sims), color='darkred', linestyle='--', linewidth=2)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Density')
    ax2.set_title(f'KDE: Match vs Distractors | AUC={auc:.3f} | Margin={margin:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Salvato plot in {output_path}")
    
    return {
        'match_mean': float(np.mean(match_sims)),
        'match_std': float(np.std(match_sims)),
        'distractor_mean': float(np.mean(top_k_distractor_sims)),
        'distractor_std': float(np.std(top_k_distractor_sims)),
        'margin': float(margin),
        'auc': float(auc)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AUDIT DELLE SEQUENZE DI POSE
# ═══════════════════════════════════════════════════════════════════════════════

def audit_pose_sequences(
    dataset: Subset,
    labels: np.ndarray,
    label_names: Dict[str, str],
    output_path: Path,
    variance_threshold: float = 1e-4,
    excursion_threshold: float = 10.0,
    max_samples: Optional[int] = None
):
    """
    Audita le sequenze di pose per identificare anomalie.
    
    Args:
        dataset: Dataset di pose
        labels: Array di etichette (può essere un subset)
        label_names: Mappatura label -> nome
        output_path: Percorso per il CSV di output
        variance_threshold: Soglia per warning quasi-static
        excursion_threshold: Soglia per warning escursione
        max_samples: Numero massimo di sample da auditare (None = tutti disponibili in labels)
    """
    print("\n" + "=" * 70)
    print("🔍 AUDIT SEQUENZE POSE")
    print("=" * 70)
    
    audit_data = []
    warnings_static = 0
    warnings_outlier = 0
    
    # Usa solo i sample per cui abbiamo le label
    n_samples = len(labels) if max_samples is None else min(max_samples, len(labels))
    
    for idx in tqdm(range(n_samples), desc="Audit pose"):
        sample = dataset[idx]
        pose = sample['pose'].cpu().numpy()  # Shape varia in base al formato
        
        # Converti a (T, 25, 3) se necessario
        if pose.ndim == 3:
            if pose.shape[0] == 3:  # (3, 25, T)
                pose = pose.transpose(2, 1, 0)
            elif pose.shape[2] == 3:  # Già (T, 25, 3)
                pass
            elif pose.shape[1] == 3:  # (T, 3, 25)
                pose = pose.transpose(0, 2, 1)
        
        # Trova frame validi (non-padding)
        frame_vars = np.var(pose, axis=(1, 2))
        valid_mask = frame_vars > 1e-8
        n_valid_frames = np.sum(valid_mask)
        
        if n_valid_frames == 0:
            # Sequenza completamente vuota
            audit_data.append({
                'idx': idx,
                'label': int(labels[idx]),
                'action_name': label_names.get(str(labels[idx]), 'Unknown'),
                'n_frames': len(pose),
                'n_valid_frames': 0,
                'x_range': 0.0,
                'y_range': 0.0,
                'z_range': 0.0,
                'mean_variance': 0.0,
                'max_variance': 0.0,
                'min_variance': 0.0,
                'warning': 'EMPTY_SEQUENCE'
            })
            warnings_static += 1
            continue
        
        valid_pose = pose[valid_mask]
        
        # Range per coordinata
        x_range = np.max(valid_pose[:, :, 0]) - np.min(valid_pose[:, :, 0])
        y_range = np.max(valid_pose[:, :, 1]) - np.min(valid_pose[:, :, 1])
        z_range = np.max(valid_pose[:, :, 2]) - np.min(valid_pose[:, :, 2])
        
        # Varianza per frame
        frame_variances = np.var(valid_pose, axis=(1, 2))
        mean_variance = np.mean(frame_variances)
        max_variance = np.max(frame_variances)
        min_variance = np.min(frame_variances)
        
        # Varianza totale sulla sequenza (movimento nel tempo)
        temporal_variance = np.var(valid_pose, axis=0).mean()
        
        # Warnings
        warning = ''
        if mean_variance < variance_threshold:
            warning = 'QUASI_STATIC'
            warnings_static += 1
        elif max(x_range, y_range, z_range) > excursion_threshold:
            warning = 'LARGE_EXCURSION'
            warnings_outlier += 1
        
        audit_data.append({
            'idx': idx,
            'label': int(labels[idx]),
            'action_name': label_names.get(str(labels[idx]), 'Unknown'),
            'n_frames': len(pose),
            'n_valid_frames': int(n_valid_frames),
            'x_range': float(x_range),
            'y_range': float(y_range),
            'z_range': float(z_range),
            'mean_variance': float(mean_variance),
            'max_variance': float(max_variance),
            'min_variance': float(min_variance),
            'temporal_variance': float(temporal_variance),
            'warning': warning
        })
    
    # Salva CSV
    df = pd.DataFrame(audit_data)
    df.to_csv(output_path, index=False)
    
    # Statistiche
    print(f"\n📈 Statistiche Audit:")
    print(f"   - Totale sequenze: {len(audit_data)}")
    print(f"   - Warning 'QUASI_STATIC' (var < {variance_threshold}): {warnings_static}")
    print(f"   - Warning 'LARGE_EXCURSION' (range > {excursion_threshold}): {warnings_outlier}")
    
    print(f"\n   📏 Range coordinate (media su tutti i sample):")
    print(f"      - X: {df['x_range'].mean():.4f} ± {df['x_range'].std():.4f}")
    print(f"      - Y: {df['y_range'].mean():.4f} ± {df['y_range'].std():.4f}")
    print(f"      - Z: {df['z_range'].mean():.4f} ± {df['z_range'].std():.4f}")
    
    print(f"\n   📊 Varianza:")
    print(f"      - Media per frame: {df['mean_variance'].mean():.6f} ± {df['mean_variance'].std():.6f}")
    print(f"      - Max per frame: {df['max_variance'].mean():.6f}")
    
    if warnings_static > 0:
        print(f"\n   ⚠️ Sample quasi-statici ({warnings_static}):")
        static_samples = df[df['warning'] == 'QUASI_STATIC'].head(10)
        for _, row in static_samples.iterrows():
            print(f"      - idx={row['idx']}, label={row['label']} ({row['action_name']}), var={row['mean_variance']:.6f}")
    
    if warnings_outlier > 0:
        print(f"\n   ⚠️ Sample con escursione eccessiva ({warnings_outlier}):")
        outlier_samples = df[df['warning'] == 'LARGE_EXCURSION'].head(10)
        for _, row in outlier_samples.iterrows():
            print(f"      - idx={row['idx']}, label={row['label']} ({row['action_name']}), "
                  f"range=({row['x_range']:.2f}, {row['y_range']:.2f}, {row['z_range']:.2f})")
    
    print(f"\n✅ Salvato audit CSV in {output_path}")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LINEAR PROBE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def linear_probe_classification(
    pose_embeds: np.ndarray,
    labels: np.ndarray,
    label_names: Dict[str, str],
    output_dir: Path,
    test_size: float = 0.3
) -> Dict:
    """
    Addestra un classificatore lineare sugli embedding di pose per valutare
    la qualità delle rappresentazioni apprese.
    
    Calcola: Top-1 accuracy, Top-5 accuracy, F1-macro, AUC, confusion matrix.
    
    Riferimento: Zhao et al. (ACCV 2022) - 90.9% accuracy su esercizi fitness.
    """
    print("\n" + "=" * 70)
    print("🎯 LINEAR PROBE CLASSIFICATION")
    print("=" * 70)
    
    # Split train/test per il linear probe
    X_train, X_test, y_train, y_test = train_test_split(
        pose_embeds, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   - Train: {len(X_train)} samples")
    print(f"   - Test:  {len(X_test)} samples")
    print(f"   - Classi: {len(np.unique(labels))}")
    
    # Addestra classificatore lineare (Logistic Regression multinomiale)
    print("\n🔧 Training Linear Classifier (Logistic Regression)...")
    clf = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Predizioni
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Top-1 Accuracy
    top1_acc = accuracy_score(y_test, y_pred)
    
    # Top-5 Accuracy (se ci sono almeno 5 classi)
    n_classes = len(np.unique(labels))
    k_top5 = min(5, n_classes)
    top5_acc = top_k_accuracy_score(y_test, y_proba, k=k_top5, labels=clf.classes_)
    
    # F1-macro
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # AUC (one-vs-rest per multiclass)
    try:
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        if y_test_bin.shape[1] == 1:  # Binary case
            auc_score = roc_auc_score(y_test_bin, y_proba[:, 1])
        else:
            auc_score = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"   ⚠️ AUC non calcolabile: {e}")
        auc_score = None
    
    # Stampa risultati
    print(f"\n📈 RISULTATI LINEAR PROBE:")
    print(f"   ┌{'─' * 35}┐")
    print(f"   │ {'Top-1 Accuracy:':<20} {top1_acc:>10.2%}  │")
    print(f"   │ {'Top-5 Accuracy:':<20} {top5_acc:>10.2%}  │")
    print(f"   │ {'F1-macro:':<20} {f1_macro:>10.4f}  │")
    if auc_score is not None:
        print(f"   │ {'AUC (macro OvR):':<20} {auc_score:>10.4f}  │")
    print(f"   └{'─' * 35}┘")
    
    # Classification Report dettagliato
    print("\n📋 Classification Report (per-class):")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    
    # Aggiungi nomi delle azioni
    report_df['action_name'] = report_df.index.map(
        lambda x: label_names.get(str(x), x) if str(x).isdigit() else x
    )
    
    # Salva report
    report_df.to_csv(output_dir / "classification_report.csv")
    
    # Stampa le classi peggiori (F1 più basso)
    class_metrics = report_df[report_df.index.astype(str).str.isdigit()].copy()
    class_metrics = class_metrics.sort_values('f1-score')
    
    print("\n   🔴 Classi con F1 più basso:")
    for idx, row in class_metrics.head(5).iterrows():
        action_name = label_names.get(str(idx), f"Class {idx}")[:30]
        print(f"      - {action_name}: F1={row['f1-score']:.3f}, "
              f"Prec={row['precision']:.3f}, Rec={row['recall']:.3f}, "
              f"Support={int(row['support'])}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(16, 14))
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=clf.classes_, yticklabels=clf.classes_)
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(clf.classes_)))
        ax.set_yticks(np.arange(len(clf.classes_)))
        ax.set_xticklabels(clf.classes_)
        ax.set_yticklabels(clf.classes_)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - Linear Probe\n'
                 f'Top-1: {top1_acc:.2%} | Top-5: {top5_acc:.2%} | F1-macro: {f1_macro:.4f}')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_linear_probe.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Trova le coppie di classi più confuse
    print("\n   🔄 Coppie di classi più confuse:")
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    # Top-10 confusioni
    flat_indices = np.argsort(cm_no_diag.flatten())[::-1][:10]
    for flat_idx in flat_indices:
        if cm_no_diag.flatten()[flat_idx] == 0:
            break
        i, j = np.unravel_index(flat_idx, cm_no_diag.shape)
        true_label = clf.classes_[i]
        pred_label = clf.classes_[j]
        true_name = label_names.get(str(true_label), f"Class {true_label}")[:25]
        pred_name = label_names.get(str(pred_label), f"Class {pred_label}")[:25]
        count = cm_no_diag[i, j]
        print(f"      - '{true_name}' → '{pred_name}': {count} errori")
    
    print(f"\n✅ Salvati report e confusion matrix in {output_dir}")
    
    return {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'f1_macro': float(f1_macro),
        'auc_macro': float(auc_score) if auc_score else None,
        'n_classes': int(n_classes),
        'n_test_samples': int(len(X_test))
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RETRIEVAL BIDIREZIONALE (Text→Pose E Pose→Text)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bidirectional_retrieval(
    pose_embeds: np.ndarray,
    text_embeds: np.ndarray,
    labels: np.ndarray,
    label_names: Dict[str, str],
    output_dir: Path
) -> Dict:
    """
    Calcola metriche di retrieval bidirezionale:
    - Text → Pose
    - Pose → Text
    
    Metriche: Recall@1, Recall@5, Recall@10, mAP, Mean Rank, Median Rank
    
    Verifica anche la simmetria del modello confrontando le due direzioni.
    """
    print("\n" + "=" * 70)
    print("🔄 RETRIEVAL BIDIREZIONALE (Text↔Pose)")
    print("=" * 70)
    
    # Normalizza embedding
    pose_norm = normalize(pose_embeds)
    text_norm = normalize(text_embeds)
    
    # Matrice di similarità (text_i, pose_j)
    sim_matrix = text_norm @ pose_norm.T
    
    results = {}
    
    for direction in ['text2pose', 'pose2text']:
        if direction == 'text2pose':
            # Query: testo, Gallery: pose
            query_sim = sim_matrix  # (N_text, N_pose)
            direction_name = "Text → Pose"
            emoji = "📝→🏃"
        else:
            # Query: pose, Gallery: testo
            query_sim = sim_matrix.T  # (N_pose, N_text)
            direction_name = "Pose → Text"
            emoji = "🏃→📝"
        
        n_queries = query_sim.shape[0]
        
        # Calcola ranks (la corrispondenza corretta è sulla diagonale)
        ranks = []
        for i in range(n_queries):
            sorted_idx = np.argsort(-query_sim[i])  # Ordine decrescente
            rank = np.where(sorted_idx == i)[0][0] + 1  # 1-based rank
            ranks.append(rank)
        ranks = np.array(ranks)
        
        # Metriche
        R1 = np.mean(ranks <= 1)
        R5 = np.mean(ranks <= 5)
        R10 = np.mean(ranks <= 10)
        mAP = np.mean([1.0 / r for r in ranks])
        mean_rank = np.mean(ranks)
        median_rank = np.median(ranks)
        
        results[direction] = {
            'recall_at_1': float(R1),
            'recall_at_5': float(R5),
            'recall_at_10': float(R10),
            'mAP': float(mAP),
            'mean_rank': float(mean_rank),
            'median_rank': float(median_rank),
            'ranks': ranks
        }
        
        print(f"\n{emoji} {direction_name}:")
        print(f"   ┌{'─' * 40}┐")
        print(f"   │ {'Recall@1:':<25} {R1:>10.2%}    │")
        print(f"   │ {'Recall@5:':<25} {R5:>10.2%}    │")
        print(f"   │ {'Recall@10:':<25} {R10:>10.2%}    │")
        print(f"   │ {'mAP:':<25} {mAP:>10.4f}    │")
        print(f"   │ {'Mean Rank:':<25} {mean_rank:>10.2f}    │")
        print(f"   │ {'Median Rank:':<25} {median_rank:>10.1f}    │")
        print(f"   └{'─' * 40}┘")
    
    # Analisi simmetria
    print("\n⚖️ ANALISI SIMMETRIA:")
    t2p = results['text2pose']
    p2t = results['pose2text']
    
    asymmetry_r1 = abs(t2p['recall_at_1'] - p2t['recall_at_1'])
    asymmetry_map = abs(t2p['mAP'] - p2t['mAP'])
    
    print(f"   - Δ Recall@1: {asymmetry_r1:.4f} ({'simmetrico' if asymmetry_r1 < 0.05 else '⚠️ asimmetrico'})")
    print(f"   - Δ mAP: {asymmetry_map:.4f} ({'simmetrico' if asymmetry_map < 0.05 else '⚠️ asimmetrico'})")
    
    # Plot confronto direzioni
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart metriche
    ax1 = axes[0]
    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'mAP']
    t2p_values = [t2p['recall_at_1'], t2p['recall_at_5'], t2p['recall_at_10'], t2p['mAP']]
    p2t_values = [p2t['recall_at_1'], p2t['recall_at_5'], p2t['recall_at_10'], p2t['mAP']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, t2p_values, width, label='Text→Pose', color='steelblue')
    bars2 = ax1.bar(x + width/2, p2t_values, width, label='Pose→Text', color='coral')
    
    ax1.set_ylabel('Score')
    ax1.set_title('Retrieval Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Annotazioni sui bar
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    # Rank distribution
    ax2 = axes[1]
    ax2.hist(t2p['ranks'], bins=50, alpha=0.6, label='Text→Pose', color='steelblue', density=True)
    ax2.hist(p2t['ranks'], bins=50, alpha=0.6, label='Pose→Text', color='coral', density=True)
    ax2.axvline(t2p['mean_rank'], color='steelblue', linestyle='--', linewidth=2, 
                label=f"Mean T→P: {t2p['mean_rank']:.1f}")
    ax2.axvline(p2t['mean_rank'], color='coral', linestyle='--', linewidth=2,
                label=f"Mean P→T: {p2t['mean_rank']:.1f}")
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Density')
    ax2.set_title('Rank Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "bidirectional_retrieval.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Salvato plot in {output_dir / 'bidirectional_retrieval.png'}")
    
    # Rimuovi ranks dall'output (troppo grande per JSON)
    for d in results.values():
        del d['ranks']
    
    results['asymmetry'] = {
        'delta_recall_at_1': float(asymmetry_r1),
        'delta_mAP': float(asymmetry_map),
        'is_symmetric': bool(asymmetry_r1 < 0.05 and asymmetry_map < 0.05)
    }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ANALISI PER CLASSE
# ═══════════════════════════════════════════════════════════════════════════════

def per_class_analysis(
    pose_embeds: np.ndarray,
    text_embeds: np.ndarray,
    labels: np.ndarray,
    label_names: Dict[str, str],
    output_dir: Path
) -> pd.DataFrame:
    """
    Analisi dettagliata delle metriche per ogni classe di esercizio/errore.
    
    Calcola per ogni classe:
    - Recall@1, Recall@5 specifici
    - Average Precision
    - Numero di sample
    - Tasso di confusione con altre classi
    
    Identifica classi problematiche e pattern di errore.
    """
    print("\n" + "=" * 70)
    print("📊 ANALISI PER CLASSE")
    print("=" * 70)
    
    # Normalizza embedding
    pose_norm = normalize(pose_embeds)
    text_norm = normalize(text_embeds)
    sim_matrix = text_norm @ pose_norm.T
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    class_metrics = []
    confusion_data = []
    
    for label in unique_labels:
        # Indici dei sample di questa classe
        class_mask = labels == label
        class_indices = np.where(class_mask)[0]
        n_samples = len(class_indices)
        
        if n_samples == 0:
            continue
        
        # Calcola ranks per questa classe
        ranks = []
        retrieved_labels = []
        
        for idx in class_indices:
            sorted_idx = np.argsort(-sim_matrix[idx])  # Ordine decrescente
            rank = np.where(sorted_idx == idx)[0][0] + 1
            ranks.append(rank)
            
            # Label del top-1 retrieved (per analisi confusione)
            top1_idx = sorted_idx[0]
            retrieved_labels.append(labels[top1_idx])
        
        ranks = np.array(ranks)
        retrieved_labels = np.array(retrieved_labels)
        
        # Metriche per classe
        R1 = np.mean(ranks <= 1)
        R5 = np.mean(ranks <= 5)
        AP = np.mean([1.0 / r for r in ranks])
        mean_rank = np.mean(ranks)
        median_rank = np.median(ranks)
        
        # Tasso di confusione (quanto spesso il top-1 ha la stessa label)
        correct_retrievals = np.sum(retrieved_labels == label)
        confusion_rate = 1 - (correct_retrievals / n_samples)
        
        # Trova la classe più confusa (escludendo la classe stessa)
        wrong_mask = retrieved_labels != label
        if np.sum(wrong_mask) > 0:
            wrong_labels = retrieved_labels[wrong_mask]
            most_confused_label = pd.Series(wrong_labels).mode()
            if len(most_confused_label) > 0:
                most_confused = int(most_confused_label.iloc[0])
                most_confused_name = label_names.get(str(most_confused), f"Class {most_confused}")
                most_confused_count = np.sum(wrong_labels == most_confused)
            else:
                most_confused = None
                most_confused_name = "N/A"
                most_confused_count = 0
        else:
            most_confused = None
            most_confused_name = "N/A"
            most_confused_count = 0
        
        action_name = label_names.get(str(label), f"Class {label}")
        
        class_metrics.append({
            'label': int(label),
            'action_name': action_name,
            'n_samples': n_samples,
            'recall_at_1': float(R1),
            'recall_at_5': float(R5),
            'average_precision': float(AP),
            'mean_rank': float(mean_rank),
            'median_rank': float(median_rank),
            'confusion_rate': float(confusion_rate),
            'most_confused_with': most_confused_name,
            'most_confused_count': int(most_confused_count)
        })
        
        # Per confusion matrix tra classi (retrieval)
        for wrong_label in np.unique(retrieved_labels[wrong_mask]):
            count = np.sum(retrieved_labels == wrong_label)
            confusion_data.append({
                'true_label': int(label),
                'true_name': action_name,
                'retrieved_label': int(wrong_label),
                'retrieved_name': label_names.get(str(wrong_label), f"Class {wrong_label}"),
                'count': int(count)
            })
    
    # DataFrame con metriche per classe
    df_metrics = pd.DataFrame(class_metrics)
    df_metrics = df_metrics.sort_values('recall_at_1')
    
    # Salva CSV
    df_metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)
    
    # Stampa classi peggiori
    print(f"\n🔴 TOP 10 CLASSI CON RECALL@1 PIÙ BASSO:")
    print(f"   {'Classe':<40} {'R@1':>8} {'R@5':>8} {'AP':>8} {'MeanR':>8} {'Samples':>8}")
    print(f"   {'-' * 88}")
    
    for _, row in df_metrics.head(10).iterrows():
        name = row['action_name'][:38]
        print(f"   {name:<40} {row['recall_at_1']:>7.1%} {row['recall_at_5']:>7.1%} "
              f"{row['average_precision']:>8.4f} {row['mean_rank']:>8.1f} {row['n_samples']:>8}")
    
    # Stampa classi con più confusioni
    print(f"\n🔄 CLASSI PIÙ CONFUSE (confusion_rate alto):")
    df_confused = df_metrics.sort_values('confusion_rate', ascending=False)
    
    for _, row in df_confused.head(5).iterrows():
        if row['confusion_rate'] > 0:
            print(f"   - {row['action_name'][:40]}: {row['confusion_rate']:.1%} confuso")
            print(f"     → Più confuso con: {row['most_confused_with']} ({row['most_confused_count']} volte)")
    
    # Statistiche aggregate
    print(f"\n📈 STATISTICHE AGGREGATE:")
    print(f"   - Recall@1 medio per classe: {df_metrics['recall_at_1'].mean():.2%}")
    print(f"   - Recall@1 std per classe: {df_metrics['recall_at_1'].std():.2%}")
    print(f"   - Classi con R@1 = 0%: {sum(df_metrics['recall_at_1'] == 0)}")
    print(f"   - Classi con R@1 = 100%: {sum(df_metrics['recall_at_1'] == 1)}")
    
    # Plot metriche per classe
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Recall@1 per classe (bar chart ordinato)
    ax1 = axes[0, 0]
    df_sorted = df_metrics.sort_values('recall_at_1', ascending=True)
    colors = ['red' if r < 0.3 else 'orange' if r < 0.7 else 'green' 
              for r in df_sorted['recall_at_1']]
    ax1.barh(range(len(df_sorted)), df_sorted['recall_at_1'], color=colors)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels([name[:20] for name in df_sorted['action_name']], fontsize=6)
    ax1.set_xlabel('Recall@1')
    ax1.set_title('Recall@1 per Classe')
    ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Distribuzione Recall@1
    ax2 = axes[0, 1]
    ax2.hist(df_metrics['recall_at_1'], bins=20, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(df_metrics['recall_at_1'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df_metrics['recall_at_1'].mean():.2%}")
    ax2.axvline(df_metrics['recall_at_1'].median(), color='orange', linestyle='--', linewidth=2,
                label=f"Median: {df_metrics['recall_at_1'].median():.2%}")
    ax2.set_xlabel('Recall@1')
    ax2.set_ylabel('Numero di Classi')
    ax2.set_title('Distribuzione Recall@1 per Classe')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Recall@1 vs Numero di Sample (scatter)
    ax3 = axes[1, 0]
    ax3.scatter(df_metrics['n_samples'], df_metrics['recall_at_1'], 
                c=df_metrics['recall_at_1'], cmap='RdYlGn', s=50, alpha=0.7)
    ax3.set_xlabel('Numero di Sample')
    ax3.set_ylabel('Recall@1')
    ax3.set_title('Recall@1 vs Dimensione Classe')
    ax3.grid(True, alpha=0.3)
    
    # Aggiungi linea di tendenza
    z = np.polyfit(df_metrics['n_samples'], df_metrics['recall_at_1'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_metrics['n_samples'].min(), df_metrics['n_samples'].max(), 100)
    ax3.plot(x_line, p(x_line), "r--", alpha=0.5, label='Trend')
    ax3.legend()
    
    # 4. Confusion Rate vs Mean Rank
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df_metrics['confusion_rate'], df_metrics['mean_rank'],
                          c=df_metrics['n_samples'], cmap='viridis', s=50, alpha=0.7)
    ax4.set_xlabel('Confusion Rate')
    ax4.set_ylabel('Mean Rank')
    ax4.set_title('Confusion Rate vs Mean Rank (color = n_samples)')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='N samples')
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_analysis.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Retrieval Confusion Matrix (classi x classi)
    if len(confusion_data) > 0:
        df_confusion = pd.DataFrame(confusion_data)
        
        # Crea matrice di confusione per retrieval
        confusion_pivot = df_confusion.pivot_table(
            index='true_label', 
            columns='retrieved_label', 
            values='count', 
            aggfunc='sum',
            fill_value=0
        )
        
        # Salva confusion data
        df_confusion.to_csv(output_dir / "retrieval_confusion_pairs.csv", index=False)
        
        # Plot heatmap se non troppo grande
        if len(unique_labels) <= 30:
            fig, ax = plt.subplots(figsize=(14, 12))
            
            if HAS_SEABORN:
                sns.heatmap(confusion_pivot, annot=True, fmt='d', cmap='Reds', ax=ax,
                            cbar_kws={'label': 'Confusion Count'})
            else:
                im = ax.imshow(confusion_pivot.values, cmap='Reds', aspect='auto')
                plt.colorbar(im, ax=ax, label='Confusion Count')
                ax.set_xticks(range(len(confusion_pivot.columns)))
                ax.set_yticks(range(len(confusion_pivot.index)))
                ax.set_xticklabels(confusion_pivot.columns)
                ax.set_yticklabels(confusion_pivot.index)
            
            ax.set_xlabel('Retrieved Label')
            ax.set_ylabel('True Label')
            ax.set_title('Retrieval Confusion Matrix (Off-diagonal errors)')
            
            plt.tight_layout()
            plt.savefig(output_dir / "retrieval_confusion_matrix.png", dpi=200, bbox_inches='tight')
            plt.close()
    
    print(f"\n✅ Salvati file in {output_dir}:")
    print(f"   - per_class_metrics.csv")
    print(f"   - per_class_analysis.png")
    print(f"   - retrieval_confusion_pairs.csv")
    
    return df_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    
    print("\n" + "═" * 70)
    print("🔬 DUAL ENCODER EMBEDDING ANALYSIS")
    print("═" * 70)
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")
    
    plots_dir, outputs_dir = setup_directories()
    print(f"📁 Output directories: {plots_dir}, {outputs_dir}")
    
    # Carica label names
    label_names = load_label_names()
    
    # Carica dati
    print(f"\n📥 Caricamento dati {'COMBINATI' if args.use_augmented else 'ORIGINALI'}...")
    data_dir = project_root / "data" / "FLAG3D"
    keypoints_data, metadata_df, annotations_dict, split = load_flag3d_data(
        use_augmented=args.use_augmented,
        data_dir=data_dir
    )
    
    # Usa validation set (dati originali anche con augmentation)
    val_indices = split["val_indices"]
    print(f"   Validation samples: {len(val_indices)}")
    
    # Costruisci dataset
    full_dataset = FLAG3DDataset(
        metadata_df=metadata_df,
        annotations_dict=annotations_dict,
        keypoints_data=keypoints_data,
        device='cpu'  # Mantieni su CPU per evitare problemi di memoria
    )
    val_dataset = Subset(full_dataset, val_indices)
    
    # Carica modelli (automaticamente dalla best epoch)
    pose_encoder, text_encoder = load_models(device=device)
    
    # Estrai embedding
    print("\n📤 Estrazione embedding...")
    pose_embeds, text_embeds, labels, metadata, indices = extract_embeddings(
        pose_encoder, text_encoder, val_dataset, device,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    print(f"   Shape pose: {pose_embeds.shape}, text: {text_embeds.shape}")
    
    # Calcola retrieval
    print("\n🔍 Calcolo retrieval metrics...")
    sim_matrix, ranks, top_k_indices = compute_retrieval_results(pose_embeds, text_embeds)
    
    # Metriche base
    R1 = np.mean(ranks <= 1)
    R5 = np.mean(ranks <= 5)
    R10 = np.mean(ranks <= 10)
    mAP = np.mean([1.0 / r for r in ranks])
    
    print(f"\n📊 METRICHE RETRIEVAL (Text → Pose):")
    print(f"   - Recall@1:  {R1:.4f} ({sum(ranks <= 1)}/{len(ranks)})")
    print(f"   - Recall@5:  {R5:.4f}")
    print(f"   - Recall@10: {R10:.4f}")
    print(f"   - mAP:       {mAP:.4f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Analisi qualitativa errori
    # ═══════════════════════════════════════════════════════════════════════════
    error_logs = analyze_retrieval_errors(
        pose_embeds, text_embeds, labels, val_dataset, label_names,
        annotations_dict, sim_matrix, ranks, top_k_indices,
        plots_dir, max_errors=args.top_k_errors
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. t-SNE / PCA visualization
    # ═══════════════════════════════════════════════════════════════════════════
    # Crea subset bilanciato per t-SNE
    pose_sub, text_sub, labels_sub, ranks_sub = create_balanced_subset(
        pose_embeds, text_embeds, labels, ranks, n_samples=args.tsne_samples
    )
    
    # t-SNE
    plot_tsne_with_matches(
        pose_sub, text_sub, labels_sub, ranks_sub, label_names,
        plots_dir / "embedding_tsne.png", method='tsne'
    )
    
    # PCA
    plot_tsne_with_matches(
        pose_sub, text_sub, labels_sub, ranks_sub, label_names,
        plots_dir / "embedding_pca.png", method='pca'
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Distribuzione similarità
    # ═══════════════════════════════════════════════════════════════════════════
    sim_stats = analyze_similarity_distribution(
        sim_matrix, labels, plots_dir / "similarity_distributions.png"
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Audit pose sequences
    # ═══════════════════════════════════════════════════════════════════════════
    audit_df = audit_pose_sequences(
        val_dataset, labels, label_names,
        outputs_dir / "pose_variance_stats.csv",
        max_samples=args.max_samples
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Linear Probe Classification
    # ═══════════════════════════════════════════════════════════════════════════
    classification_results = linear_probe_classification(
        pose_embeds, labels, label_names, plots_dir
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Retrieval Bidirezionale (Text↔Pose)
    # ═══════════════════════════════════════════════════════════════════════════
    retrieval_results = compute_bidirectional_retrieval(
        pose_embeds, text_embeds, labels, label_names, plots_dir
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Analisi Per Classe
    # ═══════════════════════════════════════════════════════════════════════════
    per_class_df = per_class_analysis(
        pose_embeds, text_embeds, labels, label_names, outputs_dir
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Summary finale
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("✅ ANALISI COMPLETATA")
    print("═" * 70)
    
    # Riepilogo metriche principali
    print("\n📊 RIEPILOGO METRICHE PRINCIPALI:")
    print(f"   ┌{'─' * 50}┐")
    print(f"   │ {'RETRIEVAL (Text→Pose):':<48} │")
    print(f"   │   Recall@1: {R1:.2%}, Recall@5: {R5:.2%}, mAP: {mAP:.4f}    │")
    print(f"   │ {'RETRIEVAL (Pose→Text):':<48} │")
    print(f"   │   Recall@1: {retrieval_results['pose2text']['recall_at_1']:.2%}, "
          f"Recall@5: {retrieval_results['pose2text']['recall_at_5']:.2%}, "
          f"mAP: {retrieval_results['pose2text']['mAP']:.4f}    │")
    print(f"   │ {'LINEAR PROBE:':<48} │")
    print(f"   │   Top-1: {classification_results['top1_accuracy']:.2%}, "
          f"Top-5: {classification_results['top5_accuracy']:.2%}, "
          f"F1-macro: {classification_results['f1_macro']:.4f}  │")
    print(f"   └{'─' * 50}┘")
    
    print(f"\n📁 File generati:")
    print(f"   - {plots_dir / 'qualitative_errors'}/*.png")
    print(f"   - {plots_dir / 'embedding_tsne.png'}")
    print(f"   - {plots_dir / 'embedding_pca.png'}")
    print(f"   - {plots_dir / 'similarity_distributions.png'}")
    print(f"   - {plots_dir / 'confusion_matrix_linear_probe.png'}")
    print(f"   - {plots_dir / 'bidirectional_retrieval.png'}")
    print(f"   - {plots_dir / 'per_class_analysis.png'}")
    print(f"   - {outputs_dir / 'retrieval_errors_log.csv'}")
    print(f"   - {outputs_dir / 'pose_variance_stats.csv'}")
    print(f"   - {outputs_dir / 'classification_report.csv'}")
    print(f"   - {outputs_dir / 'per_class_metrics.csv'}")
    
    # Salva summary JSON completo
    summary = {
        'retrieval_text2pose': {
            'recall_at_1': float(R1),
            'recall_at_5': float(R5),
            'recall_at_10': float(R10),
            'mAP': float(mAP),
            'mean_rank': float(retrieval_results['text2pose']['mean_rank']),
            'median_rank': float(retrieval_results['text2pose']['median_rank']),
            'n_samples': len(labels),
            'n_errors': int(sum(ranks > 1))
        },
        'retrieval_pose2text': retrieval_results['pose2text'],
        'retrieval_asymmetry': retrieval_results['asymmetry'],
        'linear_probe': classification_results,
        'similarity': sim_stats,
        'per_class_stats': {
            'mean_recall_at_1': float(per_class_df['recall_at_1'].mean()),
            'std_recall_at_1': float(per_class_df['recall_at_1'].std()),
            'classes_with_zero_r1': int(sum(per_class_df['recall_at_1'] == 0)),
            'classes_with_perfect_r1': int(sum(per_class_df['recall_at_1'] == 1)),
            'worst_class': per_class_df.iloc[0]['action_name'],
            'worst_class_r1': float(per_class_df.iloc[0]['recall_at_1'])
        },
        'audit': {
            'n_samples': len(audit_df),
            'n_quasi_static': int(sum(audit_df['warning'] == 'QUASI_STATIC')),
            'n_large_excursion': int(sum(audit_df['warning'] == 'LARGE_EXCURSION'))
        }
    }
    
    with open(outputs_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   - {outputs_dir / 'analysis_summary.json'}")
    print("\n🎉 Done!\n")


if __name__ == "__main__":
    main()

