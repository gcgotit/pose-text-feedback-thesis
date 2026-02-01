# Temporal Resampling Report

**Generated:** 2026-02-01T20:42:40.180928
**Target Frames:** 300

## Configuration Comparison

| Aspect | Legacy | Resampled |
|--------|--------|-----------|
| FLAG3D max_frames | 550 | 300 |
| FLAG3D aug max_frames | 300 | 300 |
| EC3D max_len | 150 | 300 |
| Preprocessing | padding with zeros / truncation | uniform temporal resampling |

## Sequence Length Statistics

### flag3d_original

- Count: 7200
- Min: 301
- Max: 3089
- Mean: 818.78
- Std: 350.39

### ec3d_original

- Count: 371
- Min: 24
- Max: 208
- Mean: 80.29
- Std: 27.94

### ec3d_resampled

- Count: 371
- Min: 300
- Max: 300
- Mean: 300.0
- Std: 0.0

## Files

### FLAG3D

| File | Exists | Size (MB) |
|------|--------|-----------|
| base_keypoints (flag3d_keypoint.pkl) | ✅ | 1688.0 |
| base_metadata (flag3d_metadata.csv) | ✅ | 0.46 |
| augmented_legacy (flag3d_keypoint_augmented.pkl) | ✅ | 2286.71 |
| augmented_resampled (flag3d_keypoint_augmented_T300_resampled.pkl) | ❌ | - |
| split_combined_resampled (flag3d_split_combined_T300_resampled.json) | ❌ | - |

### EC3D

| File | Exists | Size (MB) |
|------|--------|-----------|
| original (ec3d_sequences.pkl) | ✅ | 8.58 |
| original_no_unknown (ec3d_sequences_no_unknown.pkl) | ✅ | 8.35 |
| resampled (ec3d_sequences_T300_resampled.pkl) | ✅ | 31.9 |
| resampled_no_unknown (ec3d_sequences_no_unknown_T300_resampled.pkl) | ✅ | 31.13 |

### MODIFIED

| File | Exists | Size (MB) |
|------|--------|-----------|
| utils (utils.py) | ✅ | 0.02 |
| flag3d_dataset (flag3d_dataset.py) | ✅ | 0.01 |
| flag3d_dataset_aug (flag3d_dataset_aug.py) | ✅ | 0.01 |
| augment_script (augment_and_save_FLAG3D.py) | ✅ | 0.02 |
| ec3d_builder (build_ec3d_resampled.py) | ✅ | 0.01 |
| ec3d_config (ec3d_config.py) | ✅ | 0.01 |
| sanity_check (sanity_check_resampling.py) | ✅ | 0.01 |

## How to Use

### Generate Ec3D Resampled

```bash
# Generate EC3D resampled dataset (12 classes)
cd /home/giov/Scrivania/Tesi/pose-text-feedback-thesis
python data/EC3D/build_ec3d_resampled.py --target_frames 300

# Generate EC3D resampled dataset (11 classes, no unknown)
python data/EC3D/build_ec3d_resampled.py --target_frames 300 --no_unknown
```

### Generate Flag3D Augmented Resampled

```bash
# Generate FLAG3D augmented with resampling
cd /home/giov/Scrivania/Tesi/pose-text-feedback-thesis
python data/FLAG3D/augment_and_save_FLAG3D.py --use_resampling --target_frames 300
```

### Validate

```bash
# Run sanity check
cd /home/giov/Scrivania/Tesi/pose-text-feedback-thesis
python scripts/sanity_check_resampling.py
```

### Use In Notebooks

```bash
# In EC3D notebooks, add at the top:
USE_RESAMPLED = True  # Enable resampled mode
TARGET_FRAMES = 300

# Then use:
from ec3d_config import EC3DDataset, load_ec3d_data, USE_RESAMPLED, TARGET_FRAMES

data = load_ec3d_data(use_resampled=USE_RESAMPLED)
dataset = EC3DDataset(data['sequences'], data['labels'], indices, use_resampled=USE_RESAMPLED)
```
