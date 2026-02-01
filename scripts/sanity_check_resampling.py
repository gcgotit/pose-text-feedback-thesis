#!/usr/bin/env python3
"""
Script di validazione rapida per i dati resampled.

Questo script verifica:
1. Shape e dtype corretti per FLAG3D (base, augmented) e EC3D
2. Assenza di NaN
3. Range di valori ragionevole
4. Verifica che la velocity stream non esploda

Uso:
    python scripts/sanity_check_resampling.py

Output:
    Report di validazione a console con eventuali errori/warning.
"""

import sys
from pathlib import Path
import pickle
import json

import numpy as np

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import DEFAULT_TARGET_FRAMES, resample_sequence


def check_sequence(seq, name, expected_shape=None):
    """
    Verifica una singola sequenza.
    
    Returns:
        dict con risultati del check
    """
    results = {
        'name': name,
        'shape': seq.shape,
        'dtype': str(seq.dtype),
        'has_nan': bool(np.isnan(seq).any()),
        'has_inf': bool(np.isinf(seq).any()),
        'min': float(seq.min()),
        'max': float(seq.max()),
        'mean': float(seq.mean()),
        'std': float(seq.std()),
        'errors': [],
        'warnings': []
    }
    
    # Check dtype
    if seq.dtype != np.float32:
        results['warnings'].append(f"dtype is {seq.dtype}, expected float32")
    
    # Check NaN
    if results['has_nan']:
        results['errors'].append("Contains NaN values!")
    
    # Check Inf
    if results['has_inf']:
        results['errors'].append("Contains Inf values!")
    
    # Check shape
    if expected_shape and seq.shape != expected_shape:
        results['errors'].append(f"Shape mismatch: {seq.shape} vs expected {expected_shape}")
    
    # Check range (valori ragionevoli per coordinate 3D normalizzate)
    if abs(results['min']) > 100 or abs(results['max']) > 100:
        results['warnings'].append(f"Large values detected: range [{results['min']:.2f}, {results['max']:.2f}]")
    
    return results


def compute_velocity_stats(seq):
    """Calcola statistiche sulla velocity stream."""
    # seq shape: (T, V, C)
    if seq.ndim != 3:
        return {'error': f'Expected 3D array, got shape {seq.shape}'}
    
    # Velocity: differenza temporale
    velocity = seq[1:] - seq[:-1]
    
    return {
        'vel_min': float(velocity.min()),
        'vel_max': float(velocity.max()),
        'vel_mean': float(velocity.mean()),
        'vel_std': float(velocity.std()),
        'vel_abs_max': float(np.abs(velocity).max()),
    }


def check_flag3d_base(data_dir, target_frames):
    """Verifica FLAG3D base."""
    print("\n" + "="*60)
    print("📊 FLAG3D BASE")
    print("="*60)
    
    keypoint_path = data_dir / "flag3d_keypoint.pkl"
    if not keypoint_path.exists():
        print(f"❌ File non trovato: {keypoint_path}")
        return None
    
    with open(keypoint_path, "rb") as f:
        data = pickle.load(f)
    
    annotations = data['annotations']
    print(f"Totale sequenze: {len(annotations)}")
    
    # Controlla 3 campioni
    results = []
    for i in [0, len(annotations)//2, len(annotations)-1]:
        seq = annotations[i]['keypoint'].squeeze(0)  # (T, 25, 3)
        
        # Resample per test
        seq_resampled = resample_sequence(seq.astype(np.float32), target_frames)
        
        r = check_sequence(seq_resampled, f"sample_{i}", expected_shape=(target_frames, 25, 3))
        r['velocity'] = compute_velocity_stats(seq_resampled)
        results.append(r)
        
        status = "✅" if not r['errors'] else "❌"
        print(f"\n{status} Sample {i}:")
        print(f"   Original shape: {seq.shape}")
        print(f"   Resampled shape: {r['shape']}")
        print(f"   Range: [{r['min']:.4f}, {r['max']:.4f}]")
        print(f"   Velocity abs max: {r['velocity']['vel_abs_max']:.4f}")
        
        if r['errors']:
            for e in r['errors']:
                print(f"   ❌ ERROR: {e}")
        if r['warnings']:
            for w in r['warnings']:
                print(f"   ⚠️ WARNING: {w}")
    
    return results


def check_flag3d_augmented_resampled(data_dir, target_frames):
    """Verifica FLAG3D augmented resampled."""
    print("\n" + "="*60)
    print("📊 FLAG3D AUGMENTED RESAMPLED")
    print("="*60)
    
    suffix = f"_T{target_frames}_resampled"
    keypoint_path = data_dir / f"flag3d_keypoint_augmented{suffix}.pkl"
    
    if not keypoint_path.exists():
        print(f"⚠️ File non trovato: {keypoint_path}")
        print("   Per generarlo: python data/FLAG3D/augment_and_save_FLAG3D.py --use_resampling")
        return None
    
    with open(keypoint_path, "rb") as f:
        data = pickle.load(f)
    
    annotations = data['annotations']
    print(f"Totale sequenze augmented: {len(annotations)}")
    
    results = []
    for i in [0, min(5, len(annotations)-1)]:
        seq = annotations[i]['keypoint']
        if seq.ndim == 4:
            seq = seq.squeeze(0)
        seq = seq.astype(np.float32)
        
        r = check_sequence(seq, f"aug_sample_{i}", expected_shape=(target_frames, 25, 3))
        r['velocity'] = compute_velocity_stats(seq)
        results.append(r)
        
        status = "✅" if not r['errors'] else "❌"
        print(f"\n{status} Augmented sample {i}:")
        print(f"   Shape: {r['shape']}")
        print(f"   Range: [{r['min']:.4f}, {r['max']:.4f}]")
        print(f"   Velocity abs max: {r['velocity']['vel_abs_max']:.4f}")
        
        if r['errors']:
            for e in r['errors']:
                print(f"   ❌ ERROR: {e}")
    
    return results


def check_ec3d_resampled(data_dir, target_frames, no_unknown=False):
    """Verifica EC3D resampled."""
    suffix = "_no_unknown" if no_unknown else ""
    name = f"EC3D{' (no_unknown)' if no_unknown else ''} RESAMPLED"
    
    print("\n" + "="*60)
    print(f"📊 {name}")
    print("="*60)
    
    pkl_path = data_dir / f"ec3d_sequences{suffix}_T{target_frames}_resampled.pkl"
    
    if not pkl_path.exists():
        print(f"⚠️ File non trovato: {pkl_path}")
        print("   Per generarlo: python data/EC3D/build_ec3d_resampled.py" + 
              (" --no_unknown" if no_unknown else ""))
        return None
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    print(f"Totale sequenze: {len(sequences)}")
    
    if 'resampling_info' in data:
        info = data['resampling_info']
        print(f"Target frames: {info.get('target_frames')}")
        print(f"Method: {info.get('method')}")
    
    results = []
    for i in [0, len(sequences)//2, len(sequences)-1]:
        seq = sequences[i]
        
        r = check_sequence(seq, f"ec3d_sample_{i}", expected_shape=(target_frames, 25, 3))
        r['velocity'] = compute_velocity_stats(seq)
        results.append(r)
        
        status = "✅" if not r['errors'] else "❌"
        print(f"\n{status} Sample {i}:")
        print(f"   Shape: {r['shape']}, dtype: {r['dtype']}")
        print(f"   Range: [{r['min']:.4f}, {r['max']:.4f}]")
        print(f"   Velocity abs max: {r['velocity']['vel_abs_max']:.4f}")
        
        if r['errors']:
            for e in r['errors']:
                print(f"   ❌ ERROR: {e}")
    
    return results


def main():
    print("\n" + "="*60)
    print("🔍 SANITY CHECK - TEMPORAL RESAMPLING")
    print("="*60)
    
    target_frames = DEFAULT_TARGET_FRAMES
    print(f"\nTarget frames: {target_frames}")
    
    flag3d_dir = project_root / "data" / "FLAG3D"
    ec3d_dir = project_root / "data" / "EC3D"
    
    all_results = {}
    errors_found = False
    
    # FLAG3D base
    r = check_flag3d_base(flag3d_dir, target_frames)
    if r:
        all_results['flag3d_base'] = r
        if any(res['errors'] for res in r):
            errors_found = True
    
    # FLAG3D augmented resampled
    r = check_flag3d_augmented_resampled(flag3d_dir, target_frames)
    if r:
        all_results['flag3d_augmented_resampled'] = r
        if any(res['errors'] for res in r):
            errors_found = True
    
    # EC3D resampled
    r = check_ec3d_resampled(ec3d_dir, target_frames, no_unknown=False)
    if r:
        all_results['ec3d_resampled'] = r
        if any(res['errors'] for res in r):
            errors_found = True
    
    # EC3D no_unknown resampled
    r = check_ec3d_resampled(ec3d_dir, target_frames, no_unknown=True)
    if r:
        all_results['ec3d_no_unknown_resampled'] = r
        if any(res['errors'] for res in r):
            errors_found = True
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    for dataset_name, results in all_results.items():
        n_errors = sum(len(r['errors']) for r in results)
        n_warnings = sum(len(r['warnings']) for r in results)
        
        if n_errors > 0:
            status = "❌"
        elif n_warnings > 0:
            status = "⚠️"
        else:
            status = "✅"
        
        print(f"{status} {dataset_name}: {n_errors} errors, {n_warnings} warnings")
    
    print("\n" + "="*60)
    if errors_found:
        print("❌ VALIDATION FAILED - Errors found!")
        return 1
    else:
        print("✅ VALIDATION PASSED - All checks OK!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

