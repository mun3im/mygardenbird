# SEAbird Audio Dataset Pipeline

Complete end-to-end pipeline for creating bird audio classification datasets from Xeno-Canto recordings.

## Overview

This pipeline automates the entire process from downloading raw audio to training deep learning models:

1. **Download** recordings from Xeno-Canto
2. **Annotate** bird vocalizations with interactive GUI
3. **Extract** 3-second segments
4. **Quality control** and filtering
5. **Optimize** dataset splits (prevents data leakage)
6. **Train** CNN models for classification

## Quick Start

```bash
# 1. Create API key file
echo "YOUR_XENO_CANTO_API_KEY" > xc_key.txt

# 2. Download recordings
python Stage2_xc_dload_all_from_species_list.py --output-dir /path/to/output

# 3. Annotate segments (interactive)
python Stage5_find_segments_interactive.py --sound-dir /path/to/output

# 4. Extract annotated segments
python Stage6_extract_annotated_segments.py /path/to/output --output-dir ./segments

# 5. Quality control
python Stage7_quality_control_selection.py ./segments --qc-only

# 6. Optimized splitting (prevents data leakage)
python Stage8a_splitter_mip.py ./segments --output-dir ./dataset

# 7. Train model
python Stage9_train_seabird_multifeature.py \
    --train_dir ./dataset/train \
    --val_dir ./dataset/val \
    --test_dir ./dataset/test \
    --use_pretrained
```

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `Stage1_xc_fetch_metadata.py` | Fetch recording metadata |
| 2 | `Stage2_xc_dload_all_from_species_list.py` | Download recordings (Xeno-Canto API v3) |
| 3 | `Stage3_xc_dload_delta_by_id.py` | Download specific recordings by ID |
| 4 | `Stage4_eda_downloads.py` | Exploratory data analysis |
| 5 | `Stage5_find_segments_interactive.py` | Interactive annotation GUI |
| 6 | `Stage6_extract_annotated_segments.py` | Extract WAV segments |
| 7 | `Stage7_quality_control_selection.py` | Quality control + basic splits |
| 8a | `Stage8a_splitter_mip.py` | Optimal splits (MIP - recommended) |
| 8b | `Stage8b_splitter_genetic_algorithm.py` | Optimal splits (GA) |
| 8c | `Stage8c_splitter_simulated_annealing.py` | Optimal splits (SA) |
| 9 | `Stage9_train_seabird_multifeature.py` | Model training & validation |

## Key Features

### Stage 2: Smart Downloading
- **API v3 compatible** with auto-key loading
- **FLAC format** (lossless) with automatic MP3→FLAC conversion
- **Organized by species and quality** grades (A/B/C/D/E)

### Stage 5: Interactive Annotation
- **Visual spectrogram + waveform** display
- **Automatic blob detection** with adjustable parameters
- **Drag-and-drop** segment repositioning
- **SNR calculation** (signal-to-noise ratio)
- **Audio playback** with progress indicator

### Stage 8: Optimized Splitting
**Why Stage 8 matters:**
- **Prevents data leakage**: Same recording never appears in multiple splits
- **Source-based separation**: Groups files by XC recording ID
- **Exact ratios**: Guarantees 75%/10%/15% (train/val/test)
- **Class balance**: Maintains distribution across all species

**Three optimization methods:**

| Method | Speed | Quality | Deterministic | Use Case |
|--------|-------|---------|---------------|----------|
| **MIP** | Fast | Optimal | ✅ Yes | Production (recommended) |
| **GA** | Moderate | Near-optimal | ❌ No | Complex constraints |
| **SA** | Moderate | Near-optimal | ❌ No | Moderate datasets |

### Stage 9: Multi-feature Training
- **4 CNN architectures**: MobileNetV3, ResNet50, VGG16, EfficientNetB0
- **3 feature types**: Mel spectrogram, STFT, MFCC
- **Data augmentation**: SpecAugment for robustness
- **Pretrained weights**: ImageNet transfer learning
- **Detailed logging**: System config, hyperparameters, runtime stats

## Installation

### Requirements
```bash
# Core dependencies
pip install numpy scipy librosa soundfile requests tqdm

# Stage 5 (annotation GUI)
pip install matplotlib sounddevice

# Stage 8 (MIP optimization)
pip install pulp

# Stage 9 (training)
pip install tensorflow==2.15.* scikit-learn seaborn

# System requirement
# ffmpeg for audio conversion (install via package manager)
```

### Get API Key
1. Register at https://xeno-canto.org
2. Get API key from https://xeno-canto.org/account/profile
3. Save to `xc_key.txt`

## Configuration

Edit `species.py` and `target_species.csv` to customize target species.

## Output Structure

```
output_dir/
├── {Species name}/
│   ├── A/          # Quality A recordings
│   │   ├── xc123456.flac
│   │   └── xc123456.txt    # Annotations
│   ├── B/          # Quality B recordings
│   └── C/          # Quality C recordings
│
extracted_segments/
├── {Species name}/
│   ├── xc123456_0.wav      # First 3s segment
│   ├── xc123456_1.wav      # Second 3s segment
│   └── ...
│
dataset/
├── train/{Species name}/   # 75% of data
├── val/{Species name}/     # 10% of data
├── test/{Species name}/    # 15% of data
└── split_info.json         # Split metadata
```

## Documentation

See [`PIPELINE_OVERVIEW.md`](PIPELINE_OVERVIEW.md) for detailed documentation including:
- Complete API reference for all stages
- Advanced configuration options
- Troubleshooting guide
- Pipeline comparison (basic vs optimized)

## Examples

### Download Specific Species
```bash
python Stage2_xc_dload_all_from_species_list.py \
    --species "Common myna" "Zebra dove" \
    --quality A B \
    --output-dir /Volumes/Evo/xc-mygarden-flac
```

### Custom Split Ratios
```bash
python Stage8a_splitter_mip.py ./segments \
    --output-dir ./dataset \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### Train with Custom Parameters
```bash
python Stage9_train_seabird_multifeature.py \
    --model efficientnetb0 \
    --feature mel \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 0.0005 \
    --use_pretrained
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{seabird_pipeline_2026,
  title = {SEAbird Audio Dataset Pipeline},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/seabird-pipeline}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

- **Xeno-Canto** for providing the audio recordings and API
- **TensorFlow** and **Keras** teams for the deep learning framework
- **librosa** developers for audio processing tools

## Support

For questions or issues:
- Open an issue on GitHub
- See `PIPELINE_OVERVIEW.md` for detailed documentation
- Check existing issues for common problems

---

**Status**: Production-ready ✅

**Last Updated**: 2026-01-30

**Pipeline Version**: 1.0.0
