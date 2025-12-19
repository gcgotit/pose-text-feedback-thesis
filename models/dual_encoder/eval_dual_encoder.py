'''
Estrazione delle embedding dal modello allenato

Per prima cosa, carichiamo il modello dual encoder già addestrato e il dataset FLAG3D 
(già disponibile in memoria o su disco). 
Il modello dual encoder è composto da due parti: un pose encoder basato su 2S-AGCN e un text encoder basato su DistilBERT. 
Entrambi mappano le rispettive modalità in un vettore di dimensione 128 nello spazio latente condiviso. 
Nel dataset FLAG3D ogni esempio è una sequenza di posa 3D (25 joint) associata a una descrizione testuale dettagliata 
dell'esercizio (nome, descrizione, punti chiave, errori comuni, ecc.). Utilizzeremo lo split di validazione 
(2160 esempi su 7200 totali) per estrarre le embedding e valutare il modello, così da avere una misura affidabile delle 
prestazioni.
'''

import torch
import pandas as pd
import numpy as np
import json, pickle
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
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import Subset, DataLoader
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D

# 1. Caricamento del dataset FLAG3D (val split)
data_dir = project_root / "data" / "FLAG3D"
metadata_df = pd.read_csv(data_dir / "flag3d_metadata.csv")
with open(data_dir / "flag3d_annotations.json", "r") as f:
    annotations = json.load(f)
with open(data_dir / "flag3d_keypoint.pkl", "rb") as f:
    keypoints_data = pickle.load(f)

# Caricamento degli indici del val split
with open(data_dir / "flag3d_split.json") as f:
    split = json.load(f)
val_indices = split["val_indices"]

full_dataset = FLAG3DDataset(
    metadata_df=metadata_df,
    annotations_dict=annotations,
    keypoints_data=keypoints_data,
    device='cpu'
)

val_dataset = Subset(full_dataset, val_indices)

# 2. Creazione del DataLoader per il validation set
batch_size = 32  # Usa lo stesso batch_size del training per consistenza
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,  # IMPORTANTE: non mescolare per mantenere la corrispondenza
    num_workers=0
)

# 3. Caricamento dei modelli e pesi addestrati
pose_encoder = TwoStreamAGCN()
text_encoder = DistilBERTTextEncoder()
log_dir = project_root / "logs"

# Carica il numero dell'epoca migliore
with open(log_dir / "best_epoch.txt") as f:
    best_epoch = int(f.read().strip())

# Carica i pesi migliori
pose_encoder.load_state_dict(torch.load(log_dir / f"pose_encoder_epoch{best_epoch}.pt", map_location='cpu'))
text_encoder.load_state_dict(torch.load(log_dir / f"text_encoder_epoch{best_epoch}.pt", map_location='cpu'))

# Setta i modelli in eval mode
pose_encoder.eval()
text_encoder.eval()

# 4. Estrazione delle embedding usando DataLoader
all_pose_embeds = []
all_text_embeds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        pose_batch = batch['pose']
        input_ids_batch = batch['input_ids']
        attention_mask_batch = batch['attention_mask']
        labels_batch = batch['label']
        
        # Ottieni le embedding per tutto il batch
        z_pose_batch = pose_encoder(pose_batch)  # shape (batch_size, 128)
        z_text_batch = text_encoder(
            input_ids=input_ids_batch, 
            attention_mask=attention_mask_batch
        )  # shape (batch_size, 128)
        
        # Accumula le embedding e le label
        all_pose_embeds.append(z_pose_batch.cpu().numpy())
        all_text_embeds.append(z_text_batch.cpu().numpy())
        all_labels.append(labels_batch.cpu().numpy())

# Concatena tutti i batch
all_pose_embeds = np.concatenate(all_pose_embeds, axis=0)  # shape (N_val, 128)
all_text_embeds = np.concatenate(all_text_embeds, axis=0)  # shape (N_val, 128)
all_labels = np.concatenate(all_labels, axis=0)  # shape (N_val,)

print(f"Estratte {len(all_pose_embeds)} embedding di pose e {len(all_text_embeds)} embedding di testo.")
print(f"Dimensioni: pose_embeds {all_pose_embeds.shape}, text_embeds {all_text_embeds.shape}")

# Salvataggio in CSV con Pandas (colonne: features + label)
df_embeds = pd.DataFrame(all_pose_embeds)
df_embeds['label'] = all_labels
df_embeds.to_csv("pose_embeddings_val.csv", index=False)

# Allo stesso modo per text embeddings:
df_text_embeds = pd.DataFrame(all_text_embeds)
df_text_embeds['label'] = all_labels
df_text_embeds.to_csv("text_embeddings_val.csv", index=False)

# In alternativa, salvataggio numpy binario
np.save("pose_embeddings_val.npy", all_pose_embeds)
np.save("text_embeddings_val.npy", all_text_embeds)
# Oppure np.savez per salvare più array insieme in un unico file .npz
np.savez("embeddings_val.npz", pose=all_pose_embeds, text=all_text_embeds, labels=all_labels)

'''
Valutiamo la qualità dello spazio latente condiviso calcolando metriche di retrieval cross-modale.
'''

# Calcolo distanze tra tutti gli embedding (modalità diversa)
# dist_matrix[i,j] = distanza coseno tra testo i e pose j
dist_matrix = pairwise_distances(all_text_embeds, all_pose_embeds, metric='cosine')

# ======================================================
# 1. RETRIEVAL Text→Pose
# ======================================================
print("=== RETRIEVAL Text→Pose ===")

# Ottieni il rank degli elementi corretti per ogni query testo->posa
correct_indices = np.arange(len(all_text_embeds))  # indice j corretto per il testo i (corrispondenza 1-1)
ranks_text_to_pose = []
for i in range(len(all_text_embeds)):
    # ordina distanze della query i in modo crescente (0 = più simile)
    sorted_idx = np.argsort(dist_matrix[i])
    rank = np.where(sorted_idx == correct_indices[i])[0][0] + 1  # posizione 1-based
    ranks_text_to_pose.append(rank)
ranks_text_to_pose = np.array(ranks_text_to_pose)

# Calcola Recall@1, Recall@5, mAP per Text→Pose
R1_text_to_pose = np.mean(ranks_text_to_pose <= 1)
R5_text_to_pose = np.mean(ranks_text_to_pose <= 5)
R10_text_to_pose = np.mean(ranks_text_to_pose <= 10)
# Average Precision per query (se solo un relev: AP = 1/rank se trovato entro N, altrimenti 0)
AP_text_to_pose = [(1.0/r if r <= len(all_pose_embeds) else 0) for r in ranks_text_to_pose] 
mAP_text_to_pose = np.mean(AP_text_to_pose)

print(f"Text→Pose: R@1 = {R1_text_to_pose:.3f}, R@5 = {R5_text_to_pose:.3f}, R@10 = {R10_text_to_pose:.3f}, mAP = {mAP_text_to_pose:.3f}")

# ======================================================
# 2. RETRIEVAL Pose→Text
# ======================================================
print("\n=== RETRIEVAL Pose→Text ===")

# Per Pose→Text, dobbiamo considerare le COLONNE della matrice delle distanze
ranks_pose_to_text = []
for j in range(len(all_pose_embeds)):
    # Prendiamo la colonna j (distanze di tutti i testi dalla pose j)
    distances_to_pose_j = dist_matrix[:, j]
    
    # Ordina le distanze in modo crescente
    sorted_idx = np.argsort(distances_to_pose_j)
    
    # Trova la posizione del testo corretto (j, perché corrispondenza 1-1)
    rank = np.where(sorted_idx == j)[0][0] + 1  # posizione 1-based
    ranks_pose_to_text.append(rank)

ranks_pose_to_text = np.array(ranks_pose_to_text)

# Calcola metriche per Pose→Text
R1_pose_to_text = np.mean(ranks_pose_to_text <= 1)
R5_pose_to_text = np.mean(ranks_pose_to_text <= 5)
R10_pose_to_text = np.mean(ranks_pose_to_text <= 10)
AP_pose_to_text = [(1.0/r if r <= len(all_text_embeds) else 0) for r in ranks_pose_to_text]
mAP_pose_to_text = np.mean(AP_pose_to_text)

print(f"Pose→Text: R@1 = {R1_pose_to_text:.3f}, R@5 = {R5_pose_to_text:.3f}, R@10 = {R10_pose_to_text:.3f}, mAP = {mAP_pose_to_text:.3f}")

# ======================================================
# 3. METRICHE COMBINATE E ANALISI
# ======================================================
print("\n=== METRICHE COMBINATE ===")

# Mean Reciprocal Rank (MRR) - entrambe le direzioni
MRR_text_to_pose = np.mean(1.0 / ranks_text_to_pose)
MRR_pose_to_text = np.mean(1.0 / ranks_pose_to_text)
MRR_mean = (MRR_text_to_pose + MRR_pose_to_text) / 2

print(f"MRR Text→Pose: {MRR_text_to_pose:.3f}")
print(f"MRR Pose→Text: {MRR_pose_to_text:.3f}")
print(f"MRR Mean: {MRR_mean:.3f}")

# Media delle metriche nelle due direzioni
R1_mean = (R1_text_to_pose + R1_pose_to_text) / 2
R5_mean = (R5_text_to_pose + R5_pose_to_text) / 2
R10_mean = (R10_text_to_pose + R10_pose_to_text) / 2
mAP_mean = (mAP_text_to_pose + mAP_pose_to_text) / 2

print(f"\nR@1 Mean: {R1_mean:.3f}")
print(f"R@5 Mean: {R5_mean:.3f}")
print(f"R@10 Mean: {R10_mean:.3f}")
print(f"mAP Mean: {mAP_mean:.3f}")

# ======================================================
# 4. ANALISI DEI RANK
# ======================================================
print("\n=== ANALISI DISTRIBUZIONE RANK ===")

def analyze_rank_distribution(ranks, direction):
    print(f"\n{direction}:")
    print(f"  Media rank: {np.mean(ranks):.1f}")
    print(f"  Mediana rank: {np.median(ranks):.1f}")
    print(f"  Min rank: {np.min(ranks)}")
    print(f"  Max rank: {np.max(ranks)}")
    
    # Distribuzione percentuale
    bins = [1, 5, 10, 20, 50, 100, 200]
    prev = 0
    for b in bins:
        if b > len(ranks):
            break
        count = np.sum((ranks > prev) & (ranks <= b))
        perc = count / len(ranks) * 100
        print(f"  Rank {prev+1}-{b}: {count} ({perc:.1f}%)")
        prev = b
    
    # Fallimenti (rank > 50)
    failures = np.sum(ranks > 50)
    if failures > 0:
        print(f"  Fallimenti (rank > 50): {failures} ({failures/len(ranks)*100:.1f}%)")

analyze_rank_distribution(ranks_text_to_pose, "Text→Pose")
analyze_rank_distribution(ranks_pose_to_text, "Pose→Text")

# ======================================================
# 5. VISUALIZZAZIONE MATRICE DI SIMILARITÀ
# ======================================================
print("\n=== MATRICE DI SIMILARITÀ ===")

# Converti distanze in similarità (1 - distanza)
similarity_matrix = 1 - dist_matrix

# Mostra similarità sulla diagonale (coppie corrette)
diagonal_similarities = np.diag(similarity_matrix)
print(f"Similarità coppie corrette:")
print(f"  Media: {np.mean(diagonal_similarities):.3f}")
print(f"  Min: {np.min(diagonal_similarities):.3f}")
print(f"  Max: {np.max(diagonal_similarities):.3f}")
print(f"  Std: {np.std(diagonal_similarities):.3f}")

# Confronto con similarità medie fuori diagonale
off_diagonal_mask = ~np.eye(len(diagonal_similarities), dtype=bool)
off_diagonal_similarities = similarity_matrix[off_diagonal_mask].flatten()
print(f"\nSimilarità coppie scorrette:")
print(f"  Media: {np.mean(off_diagonal_similarities):.3f}")
print(f"  Min: {np.min(off_diagonal_similarities):.3f}")
print(f"  Max: {np.max(off_diagonal_similarities):.3f}")

# Calcola separation score
separation = np.mean(diagonal_similarities) - np.mean(off_diagonal_similarities)
print(f"\nSeparation score: {separation:.3f}")
print(f"(Valori > 0.3 indicano buona separazione)")

# ======================================================
# 6. SALVATAGGIO RISULTATI
# ======================================================
results = {
    "text_to_pose": {
        "R1": float(R1_text_to_pose),
        "R5": float(R5_text_to_pose),
        "R10": float(R10_text_to_pose),
        "mAP": float(mAP_text_to_pose),
        "MRR": float(MRR_text_to_pose),
        "mean_rank": float(np.mean(ranks_text_to_pose))
    },
    "pose_to_text": {
        "R1": float(R1_pose_to_text),
        "R5": float(R5_pose_to_text),
        "R10": float(R10_pose_to_text),
        "mAP": float(mAP_pose_to_text),
        "MRR": float(MRR_pose_to_text),
        "mean_rank": float(np.mean(ranks_pose_to_text))
    },
    "combined": {
        "R1_mean": float(R1_mean),
        "R5_mean": float(R5_mean),
        "R10_mean": float(R10_mean),
        "mAP_mean": float(mAP_mean),
        "MRR_mean": float(MRR_mean)
    }
}

# Salva in JSON
with open("retrieval_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

# Salva i rank per analisi dettagliata
rank_df = pd.DataFrame({
    "text_to_pose_rank": ranks_text_to_pose,
    "pose_to_text_rank": ranks_pose_to_text
})
rank_df.to_csv("retrieval_ranks.csv", index=False)

print("\nRisultati salvati in 'retrieval_metrics.json' e 'retrieval_ranks.csv'")

'''
Visualizzazione dello spazio latente con t-SNE e PCA
'''

# Combina pose e testo embedding per visualizzarli insieme
X = np.concatenate([all_pose_embeds, all_text_embeds], axis=0)
X = normalize(X, norm='l2')  # normalizzazione globale

y = np.concatenate([all_labels, all_labels], axis=0)  # le label rimangono le stesse per le coppie
modality = np.array([0]*len(all_pose_embeds) + [1]*len(all_text_embeds)) 

# t-SNE per ridurre a 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
X_2d = tsne.fit_transform(X)

# Plot scatter 2D: colore per classe, marker diverso per modalità
plt.figure(figsize=(8,6))
for class_id in np.unique(y):
    idx = (y == class_id)
    plt.scatter(X_2d[idx & (modality==0), 0], X_2d[idx & (modality==0), 1], 
                label=f"Azione {class_id} (pose)", marker='o', s=10, alpha=0.6)
    plt.scatter(X_2d[idx & (modality==1), 0], X_2d[idx & (modality==1), 1], 
                label=f"Azione {class_id} (testo)", marker='^', s=10, alpha=0.6)
plt.title("t-SNE delle embedding latenti (ogni azione in un colore)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# t-SNE 3D
X_3d = TSNE(n_components=3, perplexity=30, random_state=42, init='pca').fit_transform(X)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for class_id in np.unique(y):
    idx = (y == class_id)
    ax.scatter(X_3d[idx & (modality==0), 0], X_3d[idx & (modality==0), 1], X_3d[idx & (modality==0), 2],
               label=f"Azione {class_id} (pose)", marker='o', s=10, alpha=0.6)
    ax.scatter(X_3d[idx & (modality==1), 0], X_3d[idx & (modality==1), 1], X_3d[idx & (modality==1), 2],
               label=f"Azione {class_id} (text)", marker='^', s=10, alpha=0.6)

ax.set_title("t-SNE 3D space (pose + text)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Varianza spiegata dai primi 2 componenti:", pca.explained_variance_ratio_)

# Plot PCA
plt.figure(figsize=(8,6))
for class_id in np.unique(y):
    idx = (y == class_id)
    plt.scatter(X_pca[idx & (modality==0), 0], X_pca[idx & (modality==0), 1], 
                label=f"Azione {class_id} (pose)", marker='o', s=10, alpha=0.6)
    plt.scatter(X_pca[idx & (modality==1), 0], X_pca[idx & (modality==1), 1], 
                label=f"Azione {class_id} (testo)", marker='^', s=10, alpha=0.6)
plt.title("PCA delle embedding latenti (ogni azione in un colore)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("\n✅ Evaluation completata con successo!")