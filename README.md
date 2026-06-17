<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=00E5FF&center=true&vCenter=true&width=600&lines=%F0%9F%A7%A0+Brain+Tumor+Detection+AI;Deep+Learning+Classification+%2B+Grading;ResNet18+Transfer+Learning" alt="Brain Tumor Detection" />

<br>

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/Model-ResNet18-005A9C?style=for-the-badge&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/Dataset-Brain+Tumor+MRI-22c55e?style=for-the-badge&labelColor=1a1a2e">
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&labelColor=1a1a2e">

<br>

<img src="https://github-readme-stats.vercel.app/api?username=basitali08&show_icons=true&theme=radical&hide_border=true&count_private=true" width="400">

</div>

---

## What It Does

| Feature | Method | Output |
|---------|--------|--------|
| **Tumor Detection** | ResNet18 CNN | Tumor / No Tumor |
| **Tumor Classification** | 4-class softmax | Glioma / Meningioma / Pituitary / No Tumor |
| **Tumor Grading** | Clinical mapping | Grade I-IV based on type |
| **Size Estimation** | Pixel intensity analysis | Estimated mm + growth stage |
| **Risk Assessment** | Rule-based | High / Moderate / Low |

---

## Architecture

```mermaid
graph TD
    A[MRI Scan Input] --> B[Resize 224x224]
    B --> C[ResNet18 Pre-trained on ImageNet]
    C --> D[Custom Classification Head]
    D --> E[Softmax Prediction]
    E --> F{Decision}
    F -->|Tumor| G[Grading Engine]
    F -->|No Tumor| H[Healthy Result]
    G --> I[Size Estimation]
    I --> J[Risk Assessment]
```

---

## Results

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

## Dashboard Features

- **Upload MRI** — Drag & drop image analysis
- **Instant Diagnosis** — Tumor type + confidence score
- **Tumor Grading** — WHO grade based on classification
- **Size Estimation** — Pixel-based measurement
- **Stage Assessment** — Early / Moderate / Advanced
- **Probability Chart** — Class-wise prediction breakdown

---

## Quick Start

```bash
# 1. Install
pip install torch torchvision streamlit plotly pandas pillow requests

# 2. Train the model
python src/brain_tumor_pipeline.py

# 3. Launch interactive dashboard
streamlit run app.py
```

---

## Project Structure

```
brain-tumor-detection/
├── app.py                          # Streamlit dashboard
├── src/
│   └── brain_tumor_pipeline.py     # Training + evaluation pipeline
├── data/
│   ├── train/                      # 120 MRI images (30/class)
│   └── test/                       # 120 MRI images (30/class)
├── models/
│   └── best_model.pth              # Trained ResNet18 weights
├── results/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── results.json
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | PyTorch 2.0+ | Deep learning engine |
| **Architecture** | ResNet18 (ImageNet pre-trained) | Transfer learning backbone |
| **Augmentation** | Random flip, rotation, color jitter | Generalization |
| **Optimizer** | AdamW + ReduceLROnPlateau | Training |
| **UI** | Streamlit + Plotly | Interactive dashboard |
| **Data** | Hugging Face Datasets | MRI images |

---

<div align="center">

**Built with Python, PyTorch, ResNet18, Streamlit**

[![GitHub stars](https://img.shields.io/github/stars/basitali08/brain-tumor-detection?style=social)](https://github.com/basitali08/brain-tumor-detection)
[![GitHub forks](https://img.shields.io/github/forks/basitali08/brain-tumor-detection?style=social)](https://github.com/basitali08/brain-tumor-detection)

</div>

---

<p align="center">
<b>Built by Basit Ali</b> · <a href="https://github.com/basitali08">GitHub</a> · <a href="mailto:whoisbasit@gmail.com">Email</a><br>
<sub>Deep Learning for Medical Imaging · MS Data Science Portfolio</sub>
</p>