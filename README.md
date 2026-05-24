# 🧠 Brain Tumor Detection & Grading AI

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![ResNet](https://img.shields.io/badge/Model-ResNet18-005A9C)](https://pytorch.org/hub/pytorch_vision_resnet/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-Brain%20Tumor%20MRI-22c55e)](https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor)

Deep learning system for **brain tumor detection**, **classification**, **grading**, and **size estimation** from MRI scans using transfer learning with ResNet18.

---

## 🎯 What It Does

| Feature | Method | Output |
|---------|--------|--------|
| **Tumor Detection** | ResNet18 CNN | Tumor / No Tumor |
| **Tumor Classification** | 4-class softmax | Glioma / Meningioma / Pituitary / No Tumor |
| **Tumor Grading** | Clinical mapping | Grade I-IV based on type |
| **Size Estimation** | Pixel intensity analysis | Estimated mm + growth stage |
| **Risk Assessment** | Rule-based | High / Moderate / Low |

---

## 🏗 Architecture

```mermaid
graph TD
    A[MRI Scan Input] --> B[Resize 224x224]
    B --> C[ResNet18<br/>Pre-trained on ImageNet<br/>18 layers · 11.4M params]
    C --> D[Custom Classification Head<br/>Dropout → 512 → ReLU → Dropout → 4]
    D --> E[Softmax → Prediction]
    E --> F{Decision}
    F -->|Tumor| G[Grading Engine<br/>Glioma: II-IV · Meningioma: I-II · Pituitary: I]
    F -->|No Tumor| H[Healthy Result]
    G --> I[Size Estimation<br/>Pixel intensity → mm]
    I --> J[Risk Assessment<br/>High / Moderate / Low]
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **100.0%** |
| **Precision (all classes)** | 1.00 |
| **Recall (all classes)** | 1.00 |
| **F1-Score (all classes)** | 1.00 |

### Per-Class Performance

| Class | Samples | Accuracy | Grade | Risk |
|-------|:-------:|:--------:|:-----:|:----:|
| **Glioma** | 30 | 100% | II-IV | High |
| **Meningioma** | 30 | 100% | I-II | Low-Moderate |
| **Pituitary** | 30 | 100% | I | Low |
| **No Tumor** | 30 | 100% | N/A | None |

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | PyTorch 2.0+ | Deep learning engine |
| **Architecture** | ResNet18 (ImageNet pre-trained) | Transfer learning backbone |
| **Augmentation** | Random flip, rotation, color jitter | Generalization |
| **Optimizer** | AdamW + ReduceLROnPlateau | Training |
| **UI** | Streamlit + Plotly | Interactive dashboard |
| **Data** | Hugging Face Datasets | MRI images |

---

## 📁 Project Structure

```
brain-tumor-detection/
├── app.py                          # Streamlit dashboard
├── src/
│   └── brain_tumor_pipeline.py     # Training + evaluation pipeline
├── data/
│   ├── train/                      # 120 MRI images (30/class)
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── no_tumor/
│   └── test/                       # 120 MRI images (30/class)
├── models/
│   └── best_model.pth              # Trained ResNet18 weights
├── results/
│   ├── confusion_matrix.png        # Classification heatmap
│   ├── training_history.png        # Loss + accuracy curves
│   └── results.json                # Full metrics
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Install
pip install torch torchvision streamlit plotly pandas pillow requests

# 2. Train the model
python src/brain_tumor_pipeline.py

# 3. Launch interactive dashboard
streamlit run app.py
```

---

## 🖥 Dashboard Features

- **Upload MRI** — Drag & drop image analysis
- **Instant Diagnosis** — Tumor type + confidence score
- **Tumor Grading** — WHO grade based on classification
- **Size Estimation** — Pixel-based measurement
- **Stage Assessment** — Early / Moderate / Advanced
- **Probability Chart** — Class-wise prediction breakdown

---

<p align="center">
<b>Built by Basit Ali</b> · <a href="https://github.com/basitali08">GitHub</a> · <a href="mailto:whoisbasit@gmail.com">Email</a><br>
<sub>Deep Learning for Medical Imaging · MS Data Science Portfolio</sub>
</p>
