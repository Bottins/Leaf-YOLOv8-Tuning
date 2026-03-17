# Leaf Detection using YOLOv8

## Overview

An end-to-end computer vision project for automated leaf detection and quantification using YOLOv8 object detection. This system processes plant images, detects individual leaves, and exports structured predictions for downstream agricultural or botanical analysis.

## Key Features

- **Deep Learning Detection**: YOLOv8n-based model optimized for high-precision leaf detection
- **End-to-End Pipeline**: Automated workflow from raw images to CSV export
- **Production-Ready Deployment**: Includes TensorFlow Lite export for edge deployment
- **Comprehensive Evaluation**: Performance metrics (mAP, precision, recall) with visualization
- **Scalable Data Processing**: Train/validation/test split with YOLO format conversion

## Project Structure

```
Leaf_dataset/
├── Leaf/
│   ├── train/          # Training images and annotations
│   ├── test/           # Test images for inference
│   ├── model/          # Pre-trained model weights (.tflite)
│   └── train.csv       # Bounding box annotations
└── Leaf_CNN.ipynb      # Complete pipeline notebook
```

## Technical Stack

- **Framework**: Ultralytics YOLOv8
- **Deep Learning**: PyTorch, TensorFlow Lite
- **Data Processing**: Pandas, OpenCV, Pillow
- **Visualization**: Matplotlib, Seaborn

## Model Performance

- **mAP50**: 0.70
- **mAP50-95**: 0.483
- **Precision**: 0.684
- **Recall**: 0.709
- **Input Resolution**: 1024×1024
- **Inference Speed**: ~14.7ms per image (Tesla T4)

## Usage

### Quick Start (Inference Only)

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('Leaf/model/best_float32.tflite')

# Run inference
model.predict('path/to/images/', save=True, imgsz=1024, conf=0.2)
```

### Full Pipeline (Training from Scratch)

1. **Data Preparation**:
   - Place images in `Leaf/train/`
   - Provide annotations in `Leaf/train.csv` (format: `image_id, width, height, bbox`)

2. **Execute Notebook**:
   - Open `Leaf_CNN.ipynb` in Jupyter/Colab
   - Run cells sequentially for:
     - Dataset splitting (80% train, 10% val, 10% test)
     - YOLO format conversion
     - Model training (50 epochs, batch size 8)
     - Evaluation and visualization

3. **Export Results**:
   - Detection predictions → CSV (`detection_summary.csv`)
   - Model weights → TensorFlow Lite format

## Installation

```bash
pip install ultralytics opencv-python pandas matplotlib seaborn scikit-learn squarify
```

## Output Format

Exported CSV contains:

| Column | Description |
|--------|-------------|
| `image_id` | Image filename |
| `numero_rilevazioni` | Number of detected leaves |
| `altezza` | Total height of detection area (pixels) |
| `larghezza` | Total width of detection area (pixels) |

## Research Profile

- **Keywords**: Object detection, YOLOv8, precision agriculture, botanical image analysis, computer vision
- **Application Domains**: Crop monitoring, plant phenotyping, agricultural automation
- **Open Source**: Reproducible research pipeline for educational and commercial use

## Acknowledgments

- Dataset and model development conducted as part of agricultural computer vision research
- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## License

Open-source for research and educational purposes. See repository for commercial usage terms.

---

**Author**: Alessandro Bottini
**Contact**: Available via GitHub issues
**Last Updated**: March 2026
