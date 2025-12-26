# 📓 Notebook: 06_ec3d_zero_shot_retrieval.ipynb

"""
Obiettivo: Valutare le performance zero-shot del dual encoder su EC3D usando due tipi di annotazioni testuali:
1. Template generico (naturale e minimale)
2. Annotazioni stile FLAG3D (opzionali, se fornite)
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from models.dual_encoder.model import DualEncoder  # aggiorna path se necessario
from utils.eval_utils import compute_retrieval_metrics

# === CONFIG ===
DATA_PATH = Path("data/ec3d/ec3d_sequences.pkl")
LABEL_MAP = {
    0: "The subject is performing a squat with correct form.",
    1: "The subject is performing a squat with feet too wide.",
    2: "The subject is performing a squat with knees inward.",
    3: "The subject is performing a squat not deep enough.",
    4: "The subject is performing a squat with the torso leaning forward.",
    5: "The subject is performing a squat with an unknown error.",
    6: "The subject is performing lunges with correct form.",
    7: "The subject is performing lunges not deep enough.",
    8: "The subject is performing lunges with knees passing the toes.",
    9: "The subject is performing a plank with correct form.",
    10: "The subject is performing a plank with a banana back.",
    11: "The subject is performing a plank with a rolled back."
}

ENCODER_CKPT = Path("checkpoints/dual_encoder_best")  # cambia path se necessario
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD DUAL ENCODER ===
print("\n📦 Loading dual encoder weights...")
pose_encoder = torch.load(ENCODER_CKPT / "pose_encoder.pt", map_location=DEVICE)
text_encoder = torch.load(ENCODER_CKPT / "text_encoder.pt", map_location=DEVICE)
pose_encoder.eval()
text_encoder.eval()

# === LOAD EC3D POSE DATA ===
print("\n📥 Loading EC3D pose sequences...")
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

pose_seqs = data["sequences"]  # [N, T, J, C]
labels = data["labels"]        # [N]

# === ENCODE TEXT ANNOTATIONS ===
print("\n💬 Encoding textual labels...")
text_list = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
text_embeds = text_encoder.encode(text_list)  # shape: [12, D]
text_embeds = torch.tensor(text_embeds, device=DEVICE)

# === ENCODE POSE SEQUENCES ===
print("\n🏃 Encoding EC3D pose embeddings...")
pose_embeddings = []
all_labels = []

for seq, label in tqdm(zip(pose_seqs, labels), total=len(labels)):
    with torch.no_grad():
        seq_tensor = torch.tensor(seq).unsqueeze(0).to(DEVICE).float()
        embed = pose_encoder(seq_tensor)  # shape: [1, D]
        pose_embeddings.append(embed.cpu().numpy())
        all_labels.append(label)

pose_embeddings = np.vstack(pose_embeddings)  # [N, D]

# === COMPUTE COSINE SIMILARITIES ===
print("\n🔍 Performing retrieval Pose → Text")
sims = cosine_similarity(pose_embeddings, text_embeds.cpu().numpy())
preds = np.argsort(-sims, axis=1)  # descending order

top1 = (preds[:, 0] == labels).mean()
top5 = np.mean([label in row[:5] for label, row in zip(labels, preds)])

# === OUTPUT METRICS ===
print("\n🎯 ZERO-SHOT RETRIEVAL METRICS (Pose → Text):")
print(f"   - Recall@1:  {top1:.4f}")
print(f"   - Recall@5:  {top5:.4f}")

# === SAVE RESULTS ===
out_dir = Path("debug_outputs/ec3d_zero_shot")
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"gt": labels, "top1": preds[:, 0]}).to_csv(out_dir / "retrieval_results.csv", index=False)

print(f"\n✅ Saved retrieval results to {out_dir}")
