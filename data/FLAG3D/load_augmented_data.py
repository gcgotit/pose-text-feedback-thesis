"""
Helper per caricare i dati FLAG3D con o senza augmentation offline.

Questo modulo fornisce funzioni per caricare:
- Solo dati originali
- Solo dati aumentati
- Dati combinati (originali + aumentati)

Esempio di utilizzo:
    from data.FLAG3D.load_augmented_data import load_flag3d_data
    
    # Solo originali
    keypoints, metadata, annotations, split = load_flag3d_data(use_augmented=False)
    
    # Combinati (originali + aumentati)
    keypoints, metadata, annotations, split = load_flag3d_data(use_augmented=True)
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def get_data_dir() -> Path:
    """Restituisce la directory dei dati FLAG3D."""
    return Path(__file__).parent


def load_original_data(data_dir: Optional[Path] = None) -> Tuple[dict, pd.DataFrame, dict, dict]:
    """
    Carica i dati originali del dataset FLAG3D.
    
    Returns:
        tuple: (keypoints_data, metadata_df, annotations, split_data)
    """
    if data_dir is None:
        data_dir = get_data_dir()
    
    # Keypoints
    with open(data_dir / "flag3d_keypoint.pkl", "rb") as f:
        keypoints_data = pickle.load(f)
    
    # Metadata
    metadata_df = pd.read_csv(data_dir / "flag3d_metadata.csv")
    
    # Annotations
    with open(data_dir / "flag3d_annotations.json", "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    # Split
    with open(data_dir / "flag3d_split.json", "r", encoding="utf-8") as f:
        split_data = json.load(f)
    
    return keypoints_data, metadata_df, annotations, split_data


def load_augmented_data(data_dir: Optional[Path] = None) -> Tuple[dict, pd.DataFrame, dict, dict]:
    """
    Carica i dati aumentati del dataset FLAG3D.
    
    Returns:
        tuple: (keypoints_data, metadata_df, annotations, split_data)
    
    Raises:
        FileNotFoundError: se i file aumentati non esistono
    """
    if data_dir is None:
        data_dir = get_data_dir()
    
    aug_keypoints_path = data_dir / "flag3d_keypoint_augmented.pkl"
    if not aug_keypoints_path.exists():
        raise FileNotFoundError(
            f"File aumentati non trovati in {data_dir}. "
            "Esegui prima augment_and_save_FLAG3D.py per generarli."
        )
    
    # Keypoints aumentati
    with open(aug_keypoints_path, "rb") as f:
        keypoints_data = pickle.load(f)
    
    # Metadata aumentati
    metadata_df = pd.read_csv(data_dir / "flag3d_metadata_augmented.csv")
    
    # Annotations aumentate
    with open(data_dir / "flag3d_annotations_augmented.json", "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    # Split aumentato
    with open(data_dir / "flag3d_split_augmented.json", "r", encoding="utf-8") as f:
        split_data = json.load(f)
    
    return keypoints_data, metadata_df, annotations, split_data


def load_combined_data(data_dir: Optional[Path] = None) -> Tuple[dict, pd.DataFrame, dict, dict]:
    """
    Carica i dati combinati (originali + aumentati) del dataset FLAG3D.
    
    I dati aumentati vengono aggiunti SOLO al training set.
    Il validation set rimane invariato (solo dati originali).
    
    Returns:
        tuple: (combined_keypoints, combined_metadata, combined_annotations, combined_split)
    """
    if data_dir is None:
        data_dir = get_data_dir()
    
    # Carica dati originali
    orig_kp, orig_meta, orig_ann, orig_split = load_original_data(data_dir)
    
    # Carica dati aumentati
    aug_kp, aug_meta, aug_ann, aug_split = load_augmented_data(data_dir)
    
    # Combina keypoints
    combined_keypoints = {
        "annotations": orig_kp["annotations"] + aug_kp["annotations"]
    }
    
    # Combina metadata
    # Reset indici per il dataframe aumentato per partire dopo gli originali
    aug_meta_reset = aug_meta.copy()
    aug_meta_reset.index = range(len(orig_meta), len(orig_meta) + len(aug_meta))
    combined_metadata = pd.concat([orig_meta, aug_meta_reset], ignore_index=True)
    
    # Combina annotations (merge dei dizionari)
    combined_annotations = {**orig_ann, **aug_ann}
    
    # Carica split combinato (se esiste) o crealo
    combined_split_path = data_dir / "flag3d_split_combined.json"
    if combined_split_path.exists():
        with open(combined_split_path, "r", encoding="utf-8") as f:
            combined_split = json.load(f)
    else:
        # Crea split combinato manualmente
        combined_split = {
            "train_indices": orig_split["train_indices"] + aug_split["train_indices"],
            "val_indices": orig_split["val_indices"],
            "train_count": len(orig_split["train_indices"]) + len(aug_split["train_indices"]),
            "val_count": len(orig_split["val_indices"]),
        }
    
    return combined_keypoints, combined_metadata, combined_annotations, combined_split


def load_flag3d_data(
    use_augmented: bool = False,
    data_dir: Optional[Path] = None
) -> Tuple[dict, pd.DataFrame, dict, dict]:
    """
    Funzione principale per caricare i dati FLAG3D.
    
    Args:
        use_augmented: Se True, carica i dati combinati (originali + aumentati).
                      Se False, carica solo i dati originali.
        data_dir: Directory dei dati (default: directory di questo file)
    
    Returns:
        tuple: (keypoints_data, metadata_df, annotations, split_data)
    
    Esempio:
        # Nel training script
        keypoints, metadata, annotations, split = load_flag3d_data(use_augmented=True)
        
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
        
        train_df = metadata.iloc[train_indices]
        val_df = metadata.iloc[val_indices]
    """
    if use_augmented:
        return load_combined_data(data_dir)
    else:
        return load_original_data(data_dir)


def check_augmented_data_exists(data_dir: Optional[Path] = None) -> bool:
    """
    Verifica se i dati aumentati esistono.
    
    Returns:
        bool: True se i file aumentati esistono, False altrimenti
    """
    if data_dir is None:
        data_dir = get_data_dir()
    
    required_files = [
        "flag3d_keypoint_augmented.pkl",
        "flag3d_metadata_augmented.csv",
        "flag3d_annotations_augmented.json",
        "flag3d_split_augmented.json",
    ]
    
    return all((data_dir / f).exists() for f in required_files)


def get_data_stats(use_augmented: bool = False, data_dir: Optional[Path] = None) -> dict:
    """
    Restituisce statistiche sui dati.
    
    Args:
        use_augmented: Se True, statistiche sui dati combinati
        data_dir: Directory dei dati
    
    Returns:
        dict: Statistiche del dataset
    """
    keypoints, metadata, annotations, split = load_flag3d_data(use_augmented, data_dir)
    
    stats = {
        "total_samples": len(keypoints["annotations"]),
        "train_samples": len(split["train_indices"]),
        "val_samples": len(split["val_indices"]),
        "unique_actions": metadata["action_id"].nunique(),
        "unique_labels": metadata["label"].nunique(),
        "annotations_count": len(annotations),
    }
    
    # Aggiungi info specifiche per dati aumentati
    if use_augmented and "original_train_count" in split:
        stats["original_train_samples"] = split.get("original_train_count", 0)
        stats["augmented_train_samples"] = split.get("augmented_train_count", 0)
    
    return stats


if __name__ == "__main__":
    # Test del modulo
    print("Test load_augmented_data.py\n")
    
    # Test dati originali
    print("=" * 50)
    print("DATI ORIGINALI")
    print("=" * 50)
    try:
        stats = get_data_stats(use_augmented=False)
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Errore: {e}")
    
    # Test dati aumentati
    print("\n" + "=" * 50)
    print("DATI COMBINATI (originali + aumentati)")
    print("=" * 50)
    if check_augmented_data_exists():
        try:
            stats = get_data_stats(use_augmented=True)
            for key, value in stats.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Errore: {e}")
    else:
        print("  Dati aumentati non trovati.")
        print("  Esegui: python augment_and_save_FLAG3D.py")


def load_flag3d_data_resampled(
    target_frames: int = 300,
    data_dir: Optional[Path] = None
) -> Tuple[dict, pd.DataFrame, dict, dict]:
    """
    Carica i dati FLAG3D con resampling (originali + augmented resampled).
    
    I dati originali vengono resampled on-the-fly nel Dataset,
    i dati augmented sono già pre-resampled a target_frames.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    
    suffix = f"_T{target_frames}_resampled"
    
    # Carica dati originali (verranno resampled nel Dataset)
    orig_kp, orig_meta, orig_ann, orig_split = load_original_data(data_dir)
    
    # Carica dati augmented già resampled
    aug_keypoints_path = data_dir / f"flag3d_keypoint_augmented{suffix}.pkl"
    if not aug_keypoints_path.exists():
        raise FileNotFoundError(
            f"File augmented resampled non trovato: {aug_keypoints_path}\n"
            f"Esegui: python augment_and_save_FLAG3D.py --use_resampling --target_frames {target_frames}"
        )
    
    with open(aug_keypoints_path, "rb") as f:
        aug_kp = pickle.load(f)
    
    aug_meta = pd.read_csv(data_dir / f"flag3d_metadata_augmented{suffix}.csv")
    
    with open(data_dir / f"flag3d_annotations_augmented{suffix}.json", "r", encoding="utf-8") as f:
        aug_ann = json.load(f)
    
    # Combina
    combined_keypoints = {
        "annotations": orig_kp["annotations"] + aug_kp["annotations"]
    }
    
    aug_meta_reset = aug_meta.copy()
    aug_meta_reset.index = range(len(orig_meta), len(orig_meta) + len(aug_meta))
    combined_metadata = pd.concat([orig_meta, aug_meta_reset], ignore_index=True)
    
    combined_annotations = {**orig_ann, **aug_ann}
    
    # Carica split combinato resampled
    split_path = data_dir / f"flag3d_split_combined{suffix}.json"
    with open(split_path, "r", encoding="utf-8") as f:
        combined_split = json.load(f)
    
    # Aggiungi info resampling
    combined_split['use_resampling'] = True
    combined_split['target_frames'] = target_frames
    
    return combined_keypoints, combined_metadata, combined_annotations, combined_split