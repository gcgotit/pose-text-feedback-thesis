#!/usr/bin/env python3
"""
Script per generare il report sul temporal resampling.

Genera:
- reports/temporal_resampling_report.md (formato leggibile)
- reports/temporal_resampling_report.json (formato machine-readable)

Il report include:
- File creati/modificati
- Differenze di preprocessing (legacy vs resampled)
- Statistiche su lunghezze temporali
- Dimensioni file
- Istruzioni per la riproduzione

Uso:
    python reports/generate_temporal_report.py
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def get_file_info(filepath):
    """Ottiene info su un file (size, exists, checksum)."""
    filepath = Path(filepath)
    if not filepath.exists():
        return {'exists': False, 'path': str(filepath)}
    
    size_bytes = filepath.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    # Checksum MD5 (solo primi 1MB per file grandi)
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        chunk = f.read(1024 * 1024)  # 1MB
        hasher.update(chunk)
    
    return {
        'exists': True,
        'path': str(filepath),
        'size_bytes': size_bytes,
        'size_mb': round(size_mb, 2),
        'checksum_md5_partial': hasher.hexdigest()[:16],
    }


def get_sequence_stats(sequences):
    """Calcola statistiche sulle lunghezze delle sequenze."""
    if not sequences:
        return None
    
    lengths = []
    for seq in sequences:
        if hasattr(seq, 'shape'):
            lengths.append(seq.shape[0])
        elif isinstance(seq, dict) and 'keypoint' in seq:
            kp = seq['keypoint']
            if kp.ndim == 4:
                kp = kp.squeeze(0)
            lengths.append(kp.shape[0])
    
    if not lengths:
        return None
    
    return {
        'count': len(lengths),
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        'mean': round(float(np.mean(lengths)), 2),
        'std': round(float(np.std(lengths)), 2),
        'median': round(float(np.median(lengths)), 2),
    }


def load_pickle_safe(path):
    """Carica un file pickle in modo sicuro."""
    import pickle
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None


def generate_report():
    """Genera il report completo."""
    
    from utils import DEFAULT_TARGET_FRAMES
    
    target_frames = DEFAULT_TARGET_FRAMES
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'target_frames': target_frames,
        'files': {},
        'statistics': {},
        'configuration': {},
        'instructions': {},
    }
    
    flag3d_dir = project_root / "data" / "FLAG3D"
    ec3d_dir = project_root / "data" / "EC3D"
    
    # =========================================================================
    # FILE INFO
    # =========================================================================
    
    # FLAG3D files
    flag3d_files = {
        'base_keypoints': flag3d_dir / "flag3d_keypoint.pkl",
        'base_metadata': flag3d_dir / "flag3d_metadata.csv",
        'augmented_legacy': flag3d_dir / "flag3d_keypoint_augmented.pkl",
        'augmented_resampled': flag3d_dir / f"flag3d_keypoint_augmented_T{target_frames}_resampled.pkl",
        'split_combined_resampled': flag3d_dir / f"flag3d_split_combined_T{target_frames}_resampled.json",
    }
    
    # EC3D files
    ec3d_files = {
        'original': ec3d_dir / "ec3d_sequences.pkl",
        'original_no_unknown': ec3d_dir / "ec3d_sequences_no_unknown.pkl",
        'resampled': ec3d_dir / f"ec3d_sequences_T{target_frames}_resampled.pkl",
        'resampled_no_unknown': ec3d_dir / f"ec3d_sequences_no_unknown_T{target_frames}_resampled.pkl",
    }
    
    # Modified scripts/modules
    modified_files = {
        'utils': project_root / "utils.py",
        'flag3d_dataset': flag3d_dir / "flag3d_dataset.py",
        'flag3d_dataset_aug': flag3d_dir / "flag3d_dataset_aug.py",
        'augment_script': flag3d_dir / "augment_and_save_FLAG3D.py",
        'ec3d_builder': ec3d_dir / "build_ec3d_resampled.py",
        'ec3d_config': project_root / "models" / "EC3D" / "ec3d_config.py",
        'sanity_check': project_root / "scripts" / "sanity_check_resampling.py",
    }
    
    report['files']['flag3d'] = {k: get_file_info(v) for k, v in flag3d_files.items()}
    report['files']['ec3d'] = {k: get_file_info(v) for k, v in ec3d_files.items()}
    report['files']['modified'] = {k: get_file_info(v) for k, v in modified_files.items()}
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    # FLAG3D original stats
    flag3d_data = load_pickle_safe(flag3d_files['base_keypoints'])
    if flag3d_data and 'annotations' in flag3d_data:
        report['statistics']['flag3d_original'] = get_sequence_stats(flag3d_data['annotations'])
    
    # EC3D original stats
    ec3d_data = load_pickle_safe(ec3d_files['original'])
    if ec3d_data and 'sequences' in ec3d_data:
        report['statistics']['ec3d_original'] = get_sequence_stats(ec3d_data['sequences'])
    
    # EC3D resampled stats
    ec3d_res_data = load_pickle_safe(ec3d_files['resampled'])
    if ec3d_res_data and 'sequences' in ec3d_res_data:
        report['statistics']['ec3d_resampled'] = get_sequence_stats(ec3d_res_data['sequences'])
        if 'resampling_info' in ec3d_res_data:
            report['statistics']['ec3d_resampled']['resampling_info'] = ec3d_res_data['resampling_info']
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    report['configuration'] = {
        'legacy': {
            'flag3d_max_frames': 550,
            'flag3d_aug_max_frames': 300,
            'ec3d_max_len': 150,
            'preprocessing': 'padding with zeros / truncation',
        },
        'resampled': {
            'target_frames': target_frames,
            'method': 'linear interpolation (upsampling) / uniform indices (downsampling)',
            'preprocessing': 'uniform temporal resampling',
        },
    }
    
    # =========================================================================
    # INSTRUCTIONS
    # =========================================================================
    
    report['instructions'] = {
        'generate_ec3d_resampled': f"""
# Generate EC3D resampled dataset (12 classes)
cd {project_root}
python data/EC3D/build_ec3d_resampled.py --target_frames {target_frames}

# Generate EC3D resampled dataset (11 classes, no unknown)
python data/EC3D/build_ec3d_resampled.py --target_frames {target_frames} --no_unknown
""".strip(),
        
        'generate_flag3d_augmented_resampled': f"""
# Generate FLAG3D augmented with resampling
cd {project_root}
python data/FLAG3D/augment_and_save_FLAG3D.py --use_resampling --target_frames {target_frames}
""".strip(),
        
        'validate': f"""
# Run sanity check
cd {project_root}
python scripts/sanity_check_resampling.py
""".strip(),
        
        'use_in_notebooks': f"""
# In EC3D notebooks, add at the top:
USE_RESAMPLED = True  # Enable resampled mode
TARGET_FRAMES = {target_frames}

# Then use:
from ec3d_config import EC3DDataset, load_ec3d_data, USE_RESAMPLED, TARGET_FRAMES

data = load_ec3d_data(use_resampled=USE_RESAMPLED)
dataset = EC3DDataset(data['sequences'], data['labels'], indices, use_resampled=USE_RESAMPLED)
""".strip(),
    }
    
    return report


def write_markdown_report(report, output_path):
    """Scrive il report in formato Markdown."""
    
    lines = []
    lines.append("# Temporal Resampling Report")
    lines.append("")
    lines.append(f"**Generated:** {report['generated_at']}")
    lines.append(f"**Target Frames:** {report['target_frames']}")
    lines.append("")
    
    # Configuration comparison
    lines.append("## Configuration Comparison")
    lines.append("")
    lines.append("| Aspect | Legacy | Resampled |")
    lines.append("|--------|--------|-----------|")
    
    legacy = report['configuration']['legacy']
    resampled = report['configuration']['resampled']
    
    lines.append(f"| FLAG3D max_frames | {legacy['flag3d_max_frames']} | {resampled['target_frames']} |")
    lines.append(f"| FLAG3D aug max_frames | {legacy['flag3d_aug_max_frames']} | {resampled['target_frames']} |")
    lines.append(f"| EC3D max_len | {legacy['ec3d_max_len']} | {resampled['target_frames']} |")
    lines.append(f"| Preprocessing | {legacy['preprocessing']} | {resampled['preprocessing']} |")
    lines.append("")
    
    # Statistics
    lines.append("## Sequence Length Statistics")
    lines.append("")
    
    for dataset_name, stats in report['statistics'].items():
        if stats:
            lines.append(f"### {dataset_name}")
            lines.append("")
            lines.append(f"- Count: {stats.get('count', 'N/A')}")
            lines.append(f"- Min: {stats.get('min', 'N/A')}")
            lines.append(f"- Max: {stats.get('max', 'N/A')}")
            lines.append(f"- Mean: {stats.get('mean', 'N/A')}")
            lines.append(f"- Std: {stats.get('std', 'N/A')}")
            lines.append("")
    
    # Files
    lines.append("## Files")
    lines.append("")
    
    for category, files in report['files'].items():
        lines.append(f"### {category.upper()}")
        lines.append("")
        lines.append("| File | Exists | Size (MB) |")
        lines.append("|------|--------|-----------|")
        
        for name, info in files.items():
            exists = "✅" if info.get('exists', False) else "❌"
            size = info.get('size_mb', '-')
            path = Path(info.get('path', '')).name
            lines.append(f"| {name} ({path}) | {exists} | {size} |")
        
        lines.append("")
    
    # Instructions
    lines.append("## How to Use")
    lines.append("")
    
    for name, instructions in report['instructions'].items():
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        lines.append("```bash")
        lines.append(instructions)
        lines.append("```")
        lines.append("")
    
    # Write file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Markdown report: {output_path}")


def write_json_report(report, output_path):
    """Scrive il report in formato JSON."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ JSON report: {output_path}")


def main():
    print("\n" + "="*60)
    print("📊 GENERATING TEMPORAL RESAMPLING REPORT")
    print("="*60 + "\n")
    
    # Generate report
    report = generate_report()
    
    # Output paths
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    md_path = reports_dir / "temporal_resampling_report.md"
    json_path = reports_dir / "temporal_resampling_report.json"
    
    # Write reports
    write_markdown_report(report, md_path)
    write_json_report(report, json_path)
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    print(f"\nTarget frames: {report['target_frames']}")
    
    # Count existing files
    existing = sum(1 for cat in report['files'].values() 
                   for f in cat.values() if f.get('exists', False))
    total = sum(len(cat) for cat in report['files'].values())
    
    print(f"Files tracked: {existing}/{total} existing")
    
    # Check if resampled data exists
    ec3d_res = report['files']['ec3d'].get('resampled', {}).get('exists', False)
    flag3d_res = report['files']['flag3d'].get('augmented_resampled', {}).get('exists', False)
    
    print(f"\nResampled data status:")
    print(f"  - EC3D resampled: {'✅ Ready' if ec3d_res else '❌ Not generated'}")
    print(f"  - FLAG3D augmented resampled: {'✅ Ready' if flag3d_res else '❌ Not generated'}")
    
    if not ec3d_res or not flag3d_res:
        print("\n⚠️ To generate missing files, run the commands in the report.")
    
    print(f"\n📁 Reports saved to:")
    print(f"   {md_path}")
    print(f"   {json_path}")
    
    print("\n" + "="*60)
    print("✅ REPORT GENERATION COMPLETE")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

