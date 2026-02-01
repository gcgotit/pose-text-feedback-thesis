#!/usr/bin/env python3
"""
Script per espandere il dataset FLAG3D con data augmentation.

Genera nuove versioni aumentate delle sequenze di pose esistenti,
mantenendo inalterato il dataset originale e salvando i dati aumentati
in file separati.

Uso Legacy (senza resampling):
    python augment_and_save_FLAG3D.py [--num_augmentations N] [--seed S]

Uso con Resampling (raccomandato per coerenza con EC3D):
    python augment_and_save_FLAG3D.py --use_resampling --target_frames 300

Output Legacy:
    - flag3d_keypoint_augmented.pkl
    - flag3d_metadata_augmented.csv
    - flag3d_annotations_augmented.json
    - flag3d_split_augmented.json

Output con Resampling (suffisso _T{target_frames}_resampled):
    - flag3d_keypoint_augmented_T300_resampled.pkl
    - flag3d_metadata_augmented_T300_resampled.csv
    - flag3d_annotations_augmented_T300_resampled.json
    - flag3d_split_augmented_T300_resampled.json
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Importa le funzioni di augmentation esistenti
from pose_augmentation import apply_pose_augmentation

# Aggiunge la root del progetto per importare utils
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import resample_sequence, DEFAULT_TARGET_FRAMES


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


def resample_pose(
    pose_array: np.ndarray, 
    target_frames: int,
    method: str = "linear"
) -> np.ndarray:
    """
    Resampla una sequenza di pose a lunghezza fissa.
    
    Args:
        pose_array: array numpy di shape (1, T, V, 3) o (T, V, 3)
        target_frames: lunghezza target
        method: 'linear' per interpolazione, 'index' per indici uniformi
    
    Returns:
        array numpy resamplato di shape (1, target_frames, V, 3) o (target_frames, V, 3)
    """
    original_shape = pose_array.shape
    has_batch_dim = pose_array.ndim == 4
    
    if has_batch_dim:
        pose_array = pose_array.squeeze(0)  # (T, V, 3)
    
    pose_array = np.asarray(pose_array, dtype=np.float32)
    resampled = resample_sequence(pose_array, target_frames, method=method)
    
    if has_batch_dim:
        resampled = resampled[np.newaxis, ...]  # (1, target_frames, V, 3)
    
    return resampled


def augment_single_pose(pose_array: np.ndarray, num_transforms: int = 2) -> np.ndarray:
    """
    Applica data augmentation a una singola sequenza di pose.
    
    Args:
        pose_array: array numpy di shape (1, T, V, 3) o (T, V, 3)
        num_transforms: numero di trasformazioni da applicare
    
    Returns:
        array numpy aumentato della stessa shape dell'input
    """
    original_shape = pose_array.shape
    
    if pose_array.ndim == 3:
        pose_array = pose_array[np.newaxis, ...]
    elif pose_array.ndim == 4 and pose_array.shape[0] == 1:
        pass
    else:
        raise ValueError(f"Shape inaspettata: {original_shape}")
    
    pose_tensor = torch.tensor(pose_array, dtype=torch.float32)
    augmented_tensor = apply_pose_augmentation(pose_tensor, num_transforms=num_transforms)
    augmented_array = augmented_tensor.numpy()
    
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
    seed: int = 42,
    use_resampling: bool = False,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    resample_method: str = "linear"
) -> tuple:
    """
    Genera il dataset aumentato con opzione di resampling.
    
    Args:
        keypoints_data: dati keypoints originali
        metadata_df: DataFrame dei metadati
        annotations: dizionario delle annotazioni
        split_data: dati dello split train/val
        num_augmentations: numero di versioni aumentate per campione
        num_transforms: numero di trasformazioni per ogni augmentation
        seed: seed per la riproducibilità
        use_resampling: se True, applica resampling prima dell'augmentation
        target_frames: lunghezza target per resampling
        resample_method: metodo di resampling ('linear' o 'index')
    
    Returns:
        tuple: (aug_keypoints, aug_metadata_df, aug_annotations, aug_split, stats)
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
    
    # Statistiche
    original_lengths = []
    
    # Contatori
    next_aug_idx = len(keypoints_data["annotations"])
    aug_action_id_counter = 10000
    
    mode_str = f"resampled to T={target_frames}" if use_resampling else "legacy (variable length)"
    
    print(f"\n{'='*60}")
    print("GENERAZIONE DATASET AUMENTATO FLAG3D")
    print(f"{'='*60}")
    print(f"Campioni originali: {len(keypoints_data['annotations'])}")
    print(f"Campioni di training: {len(train_indices)}")
    print(f"Augmentazioni per campione: {num_augmentations}")
    print(f"Trasformazioni per augmentation: {num_transforms}")
    print(f"Modalità temporale: {mode_str}")
    if use_resampling:
        print(f"Metodo resampling: {resample_method}")
    print(f"{'='*60}\n")
    
    for idx in tqdm(train_indices, desc="Generazione augmentation"):
        row = metadata_df.iloc[idx]
        original_keypoint = keypoints_data["annotations"][idx]["keypoint"]
        original_action_id = row["action_id"]
        
        # Traccia lunghezza originale
        kp_squeezed = original_keypoint.squeeze(0) if original_keypoint.ndim == 4 else original_keypoint
        original_lengths.append(kp_squeezed.shape[0])
        
        # Opzionalmente resampla prima dell'augmentation
        if use_resampling:
            keypoint_to_augment = resample_pose(
                original_keypoint, 
                target_frames=target_frames,
                method=resample_method
            )
        else:
            keypoint_to_augment = original_keypoint
        
        # Genera N versioni aumentate
        for aug_i in range(num_augmentations):
            aug_keypoint = augment_single_pose(keypoint_to_augment, num_transforms=num_transforms)
            
            aug_keypoints_list.append({
                "keypoint": aug_keypoint,
                "original_idx": idx,
                "augmentation_idx": aug_i
            })
            
            aug_action_id = aug_action_id_counter
            aug_action_id_counter += 1
            
            # Calcola total_frames per metadata
            if use_resampling:
                total_frames_value = target_frames
            else:
                total_frames_value = int(row["total_frames"])
            
            new_row = {
                "idx": int(next_aug_idx),
                "frame_dir": create_augmented_frame_dir(row["frame_dir"], aug_i),
                "model_id": int(row["model_id"]),
                "person_id": int(row["person_id"]),
                "action_id": int(aug_action_id),
                "repeat_id": int(row["repeat_id"]),
                "label": int(row["label"]),
                "total_frames": total_frames_value,
                "action_name": row["action_name"] + f" (aug {aug_i})",
                "split": "train",
                "original_idx": int(idx),
                "augmentation_type": f"aug_{aug_i}"
            }
            aug_metadata_rows.append(new_row)
            
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
                "original_action_id": int(original_action_id)
            }
            
            aug_train_indices.append(int(next_aug_idx))
            next_aug_idx += 1
    
    aug_metadata_df = pd.DataFrame(aug_metadata_rows)
    aug_keypoints_data = {"annotations": aug_keypoints_list}
    
    aug_split = {
        "train_indices": aug_train_indices,
        "val_indices": [],
        "train_count": len(aug_train_indices),
        "val_count": 0,
        "use_resampling": use_resampling,
        "target_frames": target_frames if use_resampling else None,
        "resample_method": resample_method if use_resampling else None,
        "note": "Questo file contiene solo gli indici dei campioni aumentati."
    }
    
    # Statistiche
    stats = {
        "original_lengths": {
            "min": int(np.min(original_lengths)),
            "max": int(np.max(original_lengths)),
            "mean": float(np.mean(original_lengths)),
            "std": float(np.std(original_lengths)),
        },
        "output_length": target_frames if use_resampling else "variable",
        "num_augmented": len(aug_keypoints_list),
        "use_resampling": use_resampling,
    }
    
    return aug_keypoints_data, aug_metadata_df, aug_annotations, aug_split, stats


def get_output_suffix(use_resampling: bool, target_frames: int, custom_suffix: str = None) -> str:
    """Genera il suffisso per i file di output."""
    if custom_suffix:
        return custom_suffix
    if use_resampling:
        return f"_T{target_frames}_resampled"
    return ""


def save_augmented_data(
    data_dir: Path,
    aug_keypoints: dict,
    aug_metadata_df: pd.DataFrame,
    aug_annotations: dict,
    aug_split: dict,
    suffix: str = ""
):
    """Salva i dati aumentati in nuovi file."""
    
    keypoints_path = data_dir / f"flag3d_keypoint_augmented{suffix}.pkl"
    with open(keypoints_path, "wb") as f:
        pickle.dump(aug_keypoints, f)
    print(f"✓ Salvato: {keypoints_path}")
    
    metadata_path = data_dir / f"flag3d_metadata_augmented{suffix}.csv"
    aug_metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Salvato: {metadata_path}")
    
    annotations_path = data_dir / f"flag3d_annotations_augmented{suffix}.json"
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(aug_annotations, f, indent=2, ensure_ascii=False)
    print(f"✓ Salvato: {annotations_path}")
    
    split_path = data_dir / f"flag3d_split_augmented{suffix}.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(aug_split, f, indent=2)
    print(f"✓ Salvato: {split_path}")
    
    return [keypoints_path, metadata_path, annotations_path, split_path]


def create_combined_split(
    data_dir: Path,
    original_split: dict,
    aug_split: dict,
    suffix: str = ""
):
    """Crea un file split combinato (originali + aumentati)."""
    combined_split = {
        "train_indices": original_split["train_indices"] + aug_split["train_indices"],
        "val_indices": original_split["val_indices"],
        "train_count": original_split["train_count"] + aug_split["train_count"],
        "val_count": original_split["val_count"],
        "original_train_count": original_split["train_count"],
        "augmented_train_count": aug_split["train_count"],
        "use_resampling": aug_split.get("use_resampling", False),
        "target_frames": aug_split.get("target_frames"),
        "note": "Combina campioni originali e aumentati. Gli aumentati sono solo nel training."
    }
    
    split_path = data_dir / f"flag3d_split_combined{suffix}.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(combined_split, f, indent=2)
    print(f"✓ Salvato: {split_path}")
    
    return combined_split, split_path


def print_summary(
    original_keypoints: dict,
    original_metadata: pd.DataFrame,
    aug_keypoints: dict,
    aug_metadata: pd.DataFrame,
    aug_split: dict,
    stats: dict,
    saved_files: list
):
    """Stampa un resoconto dell'operazione di augmentation."""
    
    print(f"\n{'='*60}")
    print("RESOCONTO GENERAZIONE DATASET AUMENTATO")
    print(f"{'='*60}")
    
    print(f"\n📊 DATI ORIGINALI:")
    print(f"   - Campioni totali: {len(original_keypoints['annotations'])}")
    print(f"   - Azioni uniche: {original_metadata['action_id'].nunique()}")
    print(f"   - Label uniche: {original_metadata['label'].nunique()}")
    print(f"   - Lunghezze originali: min={stats['original_lengths']['min']}, "
          f"max={stats['original_lengths']['max']}, mean={stats['original_lengths']['mean']:.1f}")
    
    print(f"\n🔄 DATI AUMENTATI:")
    print(f"   - Campioni generati: {len(aug_keypoints['annotations'])}")
    print(f"   - Nuovi action_id: {aug_metadata['action_id'].nunique()}")
    print(f"   - Lunghezza output: {stats['output_length']}")
    print(f"   - Resampling: {'Sì' if stats['use_resampling'] else 'No (legacy)'}")
    
    print(f"\n📈 TOTALE COMBINATO:")
    total = len(original_keypoints['annotations']) + len(aug_keypoints['annotations'])
    print(f"   - Campioni totali: {total}")
    
    # Verifica shape
    sample_original = original_keypoints['annotations'][0]['keypoint']
    sample_augmented = aug_keypoints['annotations'][0]['keypoint']
    print(f"\n📐 SHAPE POSE:")
    print(f"   - Originale: {sample_original.shape}")
    print(f"   - Aumentata: {sample_augmented.shape}")
    
    print(f"\n📁 FILE SALVATI:")
    for f in saved_files:
        print(f"   - {f}")
    
    print(f"\n{'='*60}")
    print("✅ AUGMENTATION COMPLETATA CON SUCCESSO!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset FLAG3D aumentato con data augmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Modalità legacy (comportamento originale)
  python augment_and_save_FLAG3D.py

  # Con resampling a 300 frame (raccomandato)
  python augment_and_save_FLAG3D.py --use_resampling --target_frames 300

  # Con resampling a 550 frame
  python augment_and_save_FLAG3D.py --use_resampling --target_frames 550 --resample_method linear
        """
    )
    parser.add_argument(
        "--num_augmentations", type=int, default=2,
        help="Numero di versioni aumentate per ogni campione (default: 2)"
    )
    parser.add_argument(
        "--num_transforms", type=int, default=2,
        help="Numero di trasformazioni per ogni augmentation (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed per la riproducibilità (default: 42)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory dei dati FLAG3D (default: directory corrente)"
    )
    # Nuovi argomenti per resampling
    parser.add_argument(
        "--use_resampling", action="store_true",
        help="Applica resampling uniforme prima dell'augmentation"
    )
    parser.add_argument(
        "--target_frames", type=int, default=DEFAULT_TARGET_FRAMES,
        help=f"Lunghezza target per resampling (default: {DEFAULT_TARGET_FRAMES})"
    )
    parser.add_argument(
        "--resample_method", type=str, default="linear", choices=["linear", "index"],
        help="Metodo di resampling: 'linear' (interpolazione) o 'index' (indici uniformi)"
    )
    parser.add_argument(
        "--out_suffix", type=str, default=None,
        help="Suffisso personalizzato per i file di output (sovrascrive default)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent
    
    print(f"\n📁 Directory dati: {data_dir}")
    
    # Carica dati originali
    print("\n📥 Caricamento dati originali...")
    keypoints_data, metadata_df, annotations, split_data = load_original_data(data_dir)
    
    # Genera dataset aumentato
    aug_keypoints, aug_metadata_df, aug_annotations, aug_split, stats = generate_augmented_dataset(
        keypoints_data=keypoints_data,
        metadata_df=metadata_df,
        annotations=annotations,
        split_data=split_data,
        num_augmentations=args.num_augmentations,
        num_transforms=args.num_transforms,
        seed=args.seed,
        use_resampling=args.use_resampling,
        target_frames=args.target_frames,
        resample_method=args.resample_method
    )
    
    # Determina suffisso output
    suffix = get_output_suffix(args.use_resampling, args.target_frames, args.out_suffix)
    
    # Salva dati aumentati
    print("\n💾 Salvataggio dati aumentati...")
    saved_files = save_augmented_data(
        data_dir=data_dir,
        aug_keypoints=aug_keypoints,
        aug_metadata_df=aug_metadata_df,
        aug_annotations=aug_annotations,
        aug_split=aug_split,
        suffix=suffix
    )
    
    # Crea split combinato
    print("\n📋 Creazione split combinato...")
    _, combined_path = create_combined_split(data_dir, split_data, aug_split, suffix=suffix)
    saved_files.append(combined_path)
    
    # Stampa resoconto
    print_summary(
        original_keypoints=keypoints_data,
        original_metadata=metadata_df,
        aug_keypoints=aug_keypoints,
        aug_metadata=aug_metadata_df,
        aug_split=aug_split,
        stats=stats,
        saved_files=saved_files
    )


if __name__ == "__main__":
    main()
