'''
Estrazione delle embedding dal modello allenato

Per prima cosa, carichiamo il modello dual encoder già addestrato e il dataset FLAG3D 
(già disponibile in memoria o su disco). 
Il modello dual encoder è composto da due parti: un pose encoder basato su 2S-AGCN e un text encoder basato su DistilBERT. 
Entrambi mappano le rispettive modalità in un vettore di dimensione 128 nello spazio latente condiviso. 
Nel dataset FLAG3D ogni esempio è una sequenza di posa 3D (25 joint) associata a una descrizione testuale dettagliata 
dell’esercizio (nome, descrizione, punti chiave, errori comuni, ecc.). Utilizzeremo lo split di validazione 
(2160 esempi su 7200 totali) per estrarre le embedding e valutare il modello, così da avere una misura affidabile delle 
prestazioni.
'''

# fixare e mettere dataloader !!

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

from models.pose_encoder.twostream_stgcn_plus import TwoStreamSTGCNPlusEncoder
from models.text_encoder.distilbert_adapter import DistilBERTTextEncoder
from data.FLAG3D.flag3d_dataset import FLAG3DDataset
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import Subset
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
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

full_dataset = FLAG3DDataset(metadata_df=metadata_df,
                             annotations_dict=annotations,
                             keypoints_data=keypoints_data,
                             device='cpu')

val_dataset = Subset(full_dataset, val_indices)

# 2. Caricamento dei modelli e pesi addestrati
# Usa lo stesso setup del training (stgcn_plus + DistilBERT freezato con adapter)
pose_encoder = TwoStreamSTGCNPlusEncoder(
    input_dim=3,
    hidden_channels=[64, 128, 256, 256],
    output_dim=128,
    num_nodes=25,
    dropout=0.1,
    fusion_dropout=0.3
)
text_encoder = DistilBERTTextEncoder(
    freeze_bert=True,
    use_adapter=True,
    adapter_bottleneck=256
)
log_dir = project_root / "logs"

# Carica il numero dell’epoca migliore
with open(log_dir / "best_epoch.txt") as f:
    best_epoch = int(f.read().strip())

# Carica i pesi migliori
pose_encoder.load_state_dict(torch.load(log_dir / f"pose_encoder_epoch{best_epoch}.pt", map_location='cpu'))
text_encoder.load_state_dict(torch.load(log_dir / f"text_encoder_epoch{best_epoch}.pt", map_location='cpu'))

# Setta i modelli in eval mode
pose_encoder.eval(); text_encoder.eval();

# 3. Estrazione delle embedding
all_pose_embeds = []
all_text_embeds = []
all_labels = []
with torch.no_grad():
    for sample in val_dataset:   # iterare sui campioni (opzione: usare DataLoader per batch)
        pose = sample['pose'].unsqueeze(0)           # aggiungi dimensione batch = 1
        input_ids = sample['input_ids'].unsqueeze(0)
        attention_mask = sample['attention_mask'].unsqueeze(0)
        # Ottieni le embedding dalle due reti
        z_pose = pose_encoder(pose)        # shape (1, 128)
        z_text = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_pose_embeds.append(z_pose.cpu().numpy().flatten())
        all_text_embeds.append(z_text.cpu().numpy().flatten())
        all_labels.append(sample['label'])
all_pose_embeds = np.array(all_pose_embeds)   # shape (N_val, 128)
all_text_embeds = np.array(all_text_embeds)   # shape (N_val, 128)
all_labels = np.array(all_labels)
print("Estratte %d embedding di pose e %d embedding di testo." % (len(all_pose_embeds), len(all_text_embeds)))


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
Valutiamo la qualità dello spazio latente condiviso calcolando metriche di retrieval cross-modale. In questo contesto, 
le metriche ci dicono quanto bene l’embedding di una modalità trova il campione corrispondente nell’altra modalità.


Recall@K: percentuale di query il cui elemento corretto appare entro le top K posizioni nei risultati.
È una metrica binaria per query: ad esempio, Recall@1 sarà la proporzione di esempi in cui l’embedding testo corretto è 
il più vicino in assoluto alla pose query (o viceversa). 
Recall@5 invece conta successo se il match corretto è tra i primi 5 risultati restituiti.


Mean Average Precision (mAP): è la media delle Average Precision su tutte le query.
L’Average Precision di una singola query è la media delle precisioni calcolate ad ogni rank in cui un risultato rilevante 
è trovato. Nel nostro caso, dato che per ogni query c’è solo un elemento realmente rilevante (la coppia pose-testo corretta), 
l’Average Precision di una query equivale a 1/rank se il risultato corretto è presente, altrimenti 0. 
La mAP aggrega queste valori su tutte le query, fornendo un indice (0-1) che riassume sia recall che ranking. 
Un valore mAP più alto indica che, mediamente, i risultati rilevanti vengono classificati più in alto.
'''

# Calcolo distanze tra tutti gli embedding (modalità diversa)
# Esempio: dist_matrix[i,j] = distanza tra testo i e pose j
dist_matrix = pairwise_distances(all_text_embeds, all_pose_embeds, metric='cosine')
# (Usiamo la distanza coseno; alternativamente si può usare distanza Euclidea 
#  o il prodotto scalare come similarità, dato che il training era contrastivo)

# Ottieni il rank degli elementi corretti per ogni query testo->posa
correct_indices = np.arange(len(all_text_embeds))  # indice j corretto per il testo i (corrispondenza 1-1)
ranks = []
for i in range(len(all_text_embeds)):
    # ordina distanze della query i in modo crescente (0 = più simile)
    sorted_idx = np.argsort(dist_matrix[i])
    rank = np.where(sorted_idx == correct_indices[i])[0][0] + 1  # posizione 1-based
    ranks.append(rank)
ranks = np.array(ranks)

# Calcola Recall@1, Recall@5, mAP
R1 = np.mean(ranks <= 1)
R5 = np.mean(ranks <= 5)
# Average Precision per query (se solo un relev: AP = 1/rank se trovato entro N, altrimenti 0)
AP = [(1.0/r if r <= len(all_pose_embeds) else 0) for r in ranks] 
mAP = np.mean(AP)
print(f"Text->Pose: Recall@1 = {R1:.3f}, Recall@5 = {R5:.3f}, mAP = {mAP:.3f}")



'''
Visualizzazione dello spazio latente con t-SNE e PCA, è utile visualizzare le embedding in uno spazio bidimensionale, 
per capire la struttura e la separabilità semantica. Due tecniche comuni sono:

PCA (Principal Component Analysis): riduce la dimensionalità massimizzando la varianza spiegata. 
Utile per avere un’idea lineare della distribuzione dei punti.

t-SNE (t-distributed Stochastic Neighbor Embedding): proietta i dati in 2D cercando di preservare le vicinanze locali, 
spesso formando cluster di punti simili. È ottimo per visualizzare cluster non-lineari di embedding.

'''

# Combina pose e testo embedding per visualizzarli insieme
X = np.concatenate([all_pose_embeds, all_text_embeds], axis=0)
X = normalize(X, norm='l2')  # normalizzazione globale

y = np.concatenate([all_labels, all_labels], axis=0)  # le label rimangono le stesse per le coppie
modality = np.array([0]*len(all_pose_embeds) + [1]*len(all_text_embeds)) 
# 'modality' potrà aiutarci a distinguere (es. 0 = pose, 1 = testo) se vogliamo marker diversi

# t-SNE per ridurre a 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
X_2d = tsne.fit_transform(X)

# Plot scatter 2D: colore per classe, marker diverso per modalità
plt.figure(figsize=(8,6))
for class_id in np.unique(y):
    idx = (y == class_id)
    plt.scatter(X_2d[idx & (modality==0), 0], X_2d[idx & (modality==0), 1], 
                label=f"Azione {class_id} (pose)", marker='o', s=10)
    plt.scatter(X_2d[idx & (modality==1), 0], X_2d[idx & (modality==1), 1], 
                label=f"Azione {class_id} (testo)", marker='^', s=10)
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
               label=f"Azione {class_id} (pose)", marker='o', s=10)
    ax.scatter(X_3d[idx & (modality==1), 0], X_3d[idx & (modality==1), 1], X_3d[idx & (modality==1), 2],
               label=f"Azione {class_id} (text)", marker='^', s=10)

ax.set_title("t-SNE 3D space (pose + text)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Varianza spiegata dai primi 2 componenti:", pca.explained_variance_ratio_)
# Plot simile a sopra ma usando X_pca
