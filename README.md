# 🎥 Flickd - AI-Powered Fashion Detection and Matching

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-orange)](https://github.com/ultralytics/ultralytics)
[![CLIP](https://img.shields.io/badge/CLIP-OpenAI-red)](https://github.com/openai/CLIP)

<div align="center">
  <em>AI-powered fashion detection and matching system for Flickd</em>
</div>

## 📋 Overview

This project is developed for Flickd Hackathon, a new-age fashion app that brings AI-powered fashion detection to your fingertips. The system enhances Flickd's capabilities by:

- Automatically detecting fashion items in user-uploaded videos
- Matching detected items with similar products from Flickd's catalog
- Analyzing fashion styles and vibes to provide personalized recommendations
- Enabling seamless product discovery and shopping experiences

Built with cutting-edge AI technologies, this system helps Flickd users discover fashion items they love, find similar products, and get style recommendations - all through the power of computer vision and natural language processing.

## 🛠️ Technologies Used

- **Python 3.8+** — Core programming language
- **YOLOv8 (Ultralytics)** — State-of-the-art object detection
- **OpenAI CLIP** — Vision-language model for product matching
- **FAISS** — Efficient similarity search and indexing
- **PyTorch** — Deep learning framework
- **OpenCV** — Image and video processing
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **scikit-learn** — Machine learning utilities
- **FastAPI** — API development (for serving models)
- **Jupyter Notebook** — Experimentation and model training
- **Matplotlib/Seaborn** — Visualization (for analysis and plots)
- **Requests** — HTTP requests for image downloads
- **jsonschema** — Output validation

## ✨ Key Features

### 🎯 Core Capabilities
- **Video Processing**
  - Intelligent frame extraction
  - High-performance video analysis
  - Batch processing support

- **Fashion Detection**
  - YOLOv8-based object detection
  - Multi-class fashion item recognition
  - Real-time processing capabilities

- **Product Matching**
  - CLIP embeddings for semantic matching
  - FAISS indexing for fast similarity search
  - Smart caching system

- **Style Analysis**
  - Vibe classification
  - Color analysis
  - Style matching

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/adi-devv/Flickd-AI-Hackathon.git
   cd Flickd-AI-Hackathon
   ```

2. **Set Up Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 Model**
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
   ```

## 📁 Project Structure

```
Flickd/
├── main.py                      # Main processing script
├── data/
│   ├── cache/
│   │   ├── cache_metadata.pkl
│   │   ├── faiss_index.bin
│   │   ├── product_info.pkl
│   │   └── product_id_to_indices.pkl
│   ├── product_data.csv
│   └── images.csv
├── detected_frames/
│   └── detected_frame_*.jpg
├── cropped_frames/
│   └── crop_frame_*_*.jpg
├── frames/
│   └── frame_*.jpg
├── outputs/
│   ├── output_*.json
│   └── vibe_*.json
├── models/
│   └── yolov8m.pt
├── api/
│   └── app.py
├── dataset/
│   └── fashion_dataset/
├── requirements.txt
├── data.yaml
└── .gitignore
```

### Key Components

- **`main.py`**: Core processing script that handles video analysis, object detection, and product matching
- **`data/`**: Contains all data-related files and caches
  - `cache/`: Stores precomputed embeddings and indices for faster processing
  - `product_data.csv`: Product catalog with details like type, description, and tags
  - `images.csv`: Image URLs and metadata for product matching
- **`detected_frames/`**: Contains frames with detection bounding boxes
- **`cropped_frames/`**: Contains individual cropped fashion items from detected frames
  - Files are named as `crop_frame_[frame_number]_[item_class].jpg`
- **`outputs/`**: JSON files containing detection results and style classifications
- **`models/`**: Contains the YOLOv8 model weights
- **`api/`**: API implementation for serving the model
- **`dataset/`**: Training and validation datasets
- **`data.yaml`**: Configuration file for model parameters and paths

## 💻 Usage

### Basic Usage
```bash
python main.py \
    --video_path path/to/video.mp4 \
    --images_csv path/to/images.csv \
    --product_data_csv path/to/products.csv \
    --caption "Your video caption" \
    --video_id "unique_id" \
    --output_json_path path/to/output.json
```

### Input Requirements
- **Video File**: MP4 format recommended
- **Product Catalog**: CSV files with required fields
- **Optional**: Vibe taxonomy for classification

### Output Format
The system generates:
- **Visual Outputs**
  - Detected frames with bounding boxes
  - Annotated video frames

- **JSON Output**
  ```json
  {
    "detections": [
      {
        "item": "dress",
        "confidence": 0.95,
        "matches": [...],
        "vibe": "casual"
      }
    ]
  }
  ```

## ⚡ Performance

- **Speed**
  - Fast frame processing
  - Efficient similarity search
  - Optimized caching

- **Accuracy**
  - High-precision detection
  - Semantic matching
  - Style classification

### Model Training Details
- **Hardware**: NVIDIA GTX 1650
- **Training Time**: 10.5 hours
- **Epochs**: 5
- **Dataset**: [Colorful Fashion Dataset for Object Detection](https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection)
  - Used for training YOLOv8m model
  - Contains diverse fashion items with annotations

### Model Performance

#### Training Results
![Training Results](runs/detect/train3/results.png)
*Training metrics over epochs showing mAP, precision, recall, and loss*

#### Confusion Matrix
![Confusion Matrix](runs/detect/train3/confusion_matrix.png)
*Confusion matrix showing model's classification performance*

#### Precision-Recall Curves
![PR Curves](runs/detect/train3/PR_curve.png)
*Precision-Recall curves for different confidence thresholds*

#### Validation Results
![Validation Batch](runs/detect/train3/val_batch0_pred.jpg)
*Example validation batch showing model predictions*

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
