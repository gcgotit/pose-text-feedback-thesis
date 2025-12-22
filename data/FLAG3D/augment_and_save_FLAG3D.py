#!/usr/bin/env python3
"""
Script per espandere il dataset FLAG3D con data augmentation.

Genera nuove versioni aumentate delle sequenze di pose esistenti,
mantenendo inalterato il dataset originale e salvando i dati aumentati
in file separati.

Uso:
    python augment_and_save_FLAG3D.py [--num_augmentations N] [--seed S]

Output:
    - flag3d_keypoint_augmented.pkl: nuove pose aumentate
    - flag3d_metadata_augmented.csv: metadati estesi con nuovi campioni
    - flag3d_annotations_augmented.json: annotazioni estese
    - flag3d_split_augmented.json: split con nuovi campioni solo in training
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Importa le funzioni di augmentation esistenti
from pose_augmentation import apply_pose_augmentation


def load_original_data(data_dir: Path):
    """Carica i dati originali del dataset FLAG3D."""
    
    # Keypoints
    keypoints_path = data_dir / "flag3d_keypoint.pkl"
    with open(keypoints_path, "rb") as f:
        keypoints_data = pickle.load(f)
    
    # Metadata
    metadata_path = data_dir / "flag3d_metadata.csv"
    metadata_df = pd.read_csv(metadata_path)
    
    # Annotations
    annotations_path = data_dir / "flag3d_annotations.json"
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    # Split
    split_path = data_dir / "flag3d_split.json"
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    
    return keypoints_data, metadata_df, annotations, split_data


def augment_single_pose(pose_array: np.ndarray, num_transforms: int = 2) -> np.ndarray:
    """
    Applica data augmentation a una singola sequenza di pose.
    
    Args:
        pose_array: array numpy di shape (1, T, V, 3) o (T, V, 3)
        num_transforms: numero di trasformazioni da applicare
    
    Returns:
        array numpy aumentato della stessa shape dell'input
    """
    # Gestisci diverse shape di input
    original_shape = pose_array.shape
    
    if pose_array.ndim == 3:
        # (T, V, 3) -> (1, T, V, 3)
        pose_array = pose_array[np.newaxis, ...]
    elif pose_array.ndim == 4 and pose_array.shape[0] == 1:
        pass  # già (1, T, V, 3)
    else:
        raise ValueError(f"Shape inaspettata: {original_shape}")
    
    # Converti in tensor PyTorch
    pose_tensor = torch.tensor(pose_array, dtype=torch.float32)
    
    # Applica augmentation
    augmented_tensor = apply_pose_augmentation(pose_tensor, num_transforms=num_transforms)
    
    # Converti di nuovo in numpy
    augmented_array = augmented_tensor.numpy()
    
    # Ripristina la shape originale se necessario
    if len(original_shape) == 3:
        augmented_array = augmented_array.squeeze(0)
    
    return augmented_array


def create_augmented_frame_dir(original_frame_dir: str, aug_idx: int) -> str:
    """Crea un nuovo identificatore frame_dir per il campione aumentato."""
    return f"{original_frame_dir}_AUG{aug_idx:03d}"


def generate_augmented_dataset(
    keypoints_data: dict,
    metadata_df: pd.DataFrame,
    annotations: dict,
    split_data: dict,
    num_augmentations: int = 2,
    num_transforms: int = 2,
    seed: int = 42
) -> tuple:
    """
    Genera il dataset aumentato.
    
    Args:
        keypoints_data: dati keypoints originali
        metadata_df: DataFrame dei metadati
        annotations: dizionario delle annotazioni
        split_data: dati dello split train/val
        num_augmentations: numero di versioni aumentate per campione
        num_transforms: numero di trasformazioni per ogni augmentation
        seed: seed per la riproducibilità
    
    Returns:
        tuple: (aug_keypoints, aug_metadata_df, aug_annotations, aug_split)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_indices = set(split_data["train_indices"])
    
    # Strutture per i dati aumentati
    aug_keypoints_list = []
    aug_metadata_rows = []
    aug_annotations = {}
    aug_train_indices = []
    
    # Contatori
    next_aug_idx = len(keypoints_data["annotations"])  # Indice per i nuovi campioni
    aug_action_id_counter = 10000  # Offset per i nuovi action_id aumentati
    
    # Mappa action_id originale -> action_id aumentato
    action_id_aug_map = {}
    
    print(f"\n{'='*60}")
    print("GENERAZIONE DATASET AUMENTATO FLAG3D")
    print(f"{'='*60}")
    print(f"Campioni originali: {len(keypoints_data['annotations'])}")
    print(f"Campioni di training: {len(train_indices)}")
    print(f"Augmentazioni per campione: {num_augmentations}")
    print(f"Trasformazioni per augmentation: {num_transforms}")
    print(f"{'='*60}\n")
    
    # Processa solo i campioni di training
    train_samples = metadata_df[metadata_df.index.isin(train_indices)]
    
    for idx in tqdm(train_indices, desc="Generazione augmentation"):
        row = metadata_df.iloc[idx]
        original_keypoint = keypoints_data["annotations"][idx]["keypoint"]
        original_action_id = row["action_id"]
        
        # Genera N versioni aumentate per ogni campione
        for aug_i in range(num_augmentations):
            # Augmenta la posa
            aug_keypoint = augment_single_pose(original_keypoint, num_transforms=num_transforms)
            
            # Crea nuova entry keypoints
            aug_keypoints_list.append({
                "keypoint": aug_keypoint,
                "original_idx": idx,
                "augmentation_idx": aug_i
            })
            
            # Crea nuovo action_id per il campione aumentato
            aug_action_id = aug_action_id_counter
            aug_action_id_counter += 1
            
            # Crea nuova riga metadata (converti tipi numpy in Python nativi)
            new_row = {
                "idx": int(next_aug_idx),
                "frame_dir": create_augmented_frame_dir(row["frame_dir"], aug_i),
                "model_id": int(row["model_id"]),
                "person_id": int(row["person_id"]),
                "action_id": int(aug_action_id),  # Nuovo action_id
                "repeat_id": int(row["repeat_id"]),
                "label": int(row["label"]),  # Stessa label dell'originale
                "total_frames": int(row["total_frames"]),
                "action_name": row["action_name"] + f" (aug {aug_i})",
                "split": "train",  # I campioni aumentati vanno solo in training
                "original_idx": int(idx),  # Riferimento al campione originale
                "augmentation_type": f"aug_{aug_i}"
            }
            aug_metadata_rows.append(new_row)
            
            # Crea nuova annotazione (copia dell'originale con nuovo action_id)
            original_annotation = annotations[str(original_action_id)]
            aug_annotations[str(aug_action_id)] = {
                "action_id": int(aug_action_id),
                "label": int(original_annotation["label"]),
                "name": original_annotation["name"],
                "description": original_annotation.get("description", ""),
                "breathing": original_annotation.get("breathing", ""),
                "key_points": original_annotation.get("key_points", ""),
                "mistakes": original_annotation.get("mistakes", ""),
                "notes": original_annotation.get("notes", ""),
                "original_action_id": int(original_action_id)  # Riferimento
            }
            
            # Aggiungi all'indice di training
            aug_train_indices.append(int(next_aug_idx))
            
            next_aug_idx += 1
    
    # Crea DataFrame metadata aumentato
    aug_metadata_df = pd.DataFrame(aug_metadata_rows)
    
    # Crea struttura keypoints aumentata
    aug_keypoints_data = {
        "annotations": aug_keypoints_list
    }
    
    # Crea nuovo split (solo augmented in training)
    aug_split = {
        "train_indices": aug_train_indices,
        "val_indices": [],  # I campioni aumentati non vanno in validation
        "train_count": len(aug_train_indices),
        "val_count": 0,
        "note": "Questo file contiene solo gli indici dei campioni aumentati. "
                "Gli indici partono dal numero di campioni originali."
    }
    
    return aug_keypoints_data, aug_metadata_df, aug_annotations, aug_split


def save_augmented_data(
    data_dir: Path,
    aug_keypoints: dict,
    aug_metadata_df: pd.DataFrame,
    aug_annotations: dict,
    aug_split: dict
):
    """Salva i dati aumentati in nuovi file."""
    
    # Keypoints aumentati
    keypoints_path = data_dir / "flag3d_keypoint_augmented.pkl"
    with open(keypoints_path, "wb") as f:
        pickle.dump(aug_keypoints, f)
    print(f"✓ Salvato: {keypoints_path}")
    
    # Metadata aumentati
    metadata_path = data_dir / "flag3d_metadata_augmented.csv"
    aug_metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Salvato: {metadata_path}")
    
    # Annotations aumentate
    annotations_path = data_dir / "flag3d_annotations_augmented.json"
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(aug_annotations, f, indent=2, ensure_ascii=False)
    print(f"✓ Salvato: {annotations_path}")
    
    # Split aumentato
    split_path = data_dir / "flag3d_split_augmented.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(aug_split, f, indent=2)
    print(f"✓ Salvato: {split_path}")


def create_combined_split(
    data_dir: Path,
    original_split: dict,
    aug_split: dict
):
    """
    Crea un file split combinato che include sia i campioni originali
    che quelli aumentati, con i campioni aumentati solo nel training.
    """
    combined_split = {
        "train_indices": original_split["train_indices"] + aug_split["train_indices"],
        "val_indices": original_split["val_indices"],  # Solo originali
        "train_count": original_split["train_count"] + aug_split["train_count"],
        "val_count": original_split["val_count"],
        "original_train_count": original_split["train_count"],
        "augmented_train_count": aug_split["train_count"],
        "note": "Questo file combina campioni originali e aumentati. "
                "Gli aumentati sono solo nel training set."
    }
    
    split_path = data_dir / "flag3d_split_combined.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(combined_split, f, indent=2)
    print(f"✓ Salvato: {split_path}")
    
    return combined_split


def print_summary(
    original_keypoints: dict,
    original_metadata: pd.DataFrame,
    aug_keypoints: dict,
    aug_metadata: pd.DataFrame,
    aug_split: dict
):
    """Stampa un resoconto dell'operazione di augmentation."""
    
    print(f"\n{'='*60}")
    print("RESOCONTO GENERAZIONE DATASET AUMENTATO")
    print(f"{'='*60}")
    
    print(f"\n📊 DATI ORIGINALI:")
    print(f"   - Campioni totali: {len(original_keypoints['annotations'])}")
    print(f"   - Azioni uniche: {original_metadata['action_id'].nunique()}")
    print(f"   - Label uniche: {original_metadata['label'].nunique()}")
    
    print(f"\n🔄 DATI AUMENTATI:")
    print(f"   - Campioni generati: {len(aug_keypoints['annotations'])}")
    print(f"   - Nuovi action_id: {aug_metadata['action_id'].nunique()}")
    
    print(f"\n📈 TOTALE COMBINATO:")
    total_samples = len(original_keypoints['annotations']) + len(aug_keypoints['annotations'])
    print(f"   - Campioni totali: {total_samples}")
    print(f"   - Training (originali + augmented): {aug_split['train_count'] + len(original_metadata[original_metadata['split'] == 'train'])}")
    
    # Verifica shape delle pose
    sample_original = original_keypoints['annotations'][0]['keypoint']
    sample_augmented = aug_keypoints['annotations'][0]['keypoint']
    print(f"\n📐 SHAPE POSE:")
    print(f"   - Originale: {sample_original.shape}")
    print(f"   - Aumentata: {sample_augmented.shape}")
    
    print(f"\n{'='*60}")
    print("✅ AUGMENTATION COMPLETATA CON SUCCESSO!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset FLAG3D aumentato con data augmentation."
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=2,
        help="Numero di versioni aumentate per ogni campione (default: 2)"
    )
    parser.add_argument(
        "--num_transforms",
        type=int,
        default=2,
        help="Numero di trasformazioni da applicare per ogni augmentation (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per la riproducibilità (default: 42)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory dei dati FLAG3D (default: directory corrente)"
    )
    
    args = parser.parse_args()
    
    # Determina la directory dei dati
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent
    
    print(f"\n📁 Directory dati: {data_dir}")
    
    # Carica dati originali
    print("\n📥 Caricamento dati originali...")
    keypoints_data, metadata_df, annotations, split_data = load_original_data(data_dir)
    
    # Genera dataset aumentato
    aug_keypoints, aug_metadata_df, aug_annotations, aug_split = generate_augmented_dataset(
        keypoints_data=keypoints_data,
        metadata_df=metadata_df,
        annotations=annotations,
        split_data=split_data,
        num_augmentations=args.num_augmentations,
        num_transforms=args.num_transforms,
        seed=args.seed
    )
    
    # Salva dati aumentati
    print("\n💾 Salvataggio dati aumentati...")
    save_augmented_data(
        data_dir=data_dir,
        aug_keypoints=aug_keypoints,
        aug_metadata_df=aug_metadata_df,
        aug_annotations=aug_annotations,
        aug_split=aug_split
    )
    
    # Crea split combinato
    print("\n📋 Creazione split combinato...")
    create_combined_split(data_dir, split_data, aug_split)
    
    # Stampa resoconto
    print_summary(
        original_keypoints=keypoints_data,
        original_metadata=metadata_df,
        aug_keypoints=aug_keypoints,
        aug_metadata=aug_metadata_df,
        aug_split=aug_split
    )


if __name__ == "__main__":
    main()

