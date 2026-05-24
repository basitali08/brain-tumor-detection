import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, sys
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLASS_LABELS = {
    "glioma": {"display": "Glioma", "grade": "Grade II-IV", "desc": "Aggressive brain tumor", "risk": "High"},
    "meningioma": {"display": "Meningioma", "grade": "Grade I-II", "desc": "Usually benign", "risk": "Low-Moderate"},
    "pituitary": {"display": "Pituitary Adenoma", "grade": "Grade I", "desc": "Benign", "risk": "Low"},
    "no_tumor": {"display": "No Tumor", "grade": "N/A", "desc": "Healthy", "risk": "None"},
}
CLASS_EMOJIS = {"glioma": "🔴", "meningioma": "🟡", "pituitary": "🟢", "no_tumor": "✅"}
COLORS = {"glioma": "#ef4444", "meningioma": "#f59e0b", "pituitary": "#22c55e", "no_tumor": "#3b82f6"}

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 4),
    )
    model_path = os.path.join(PROJECT_DIR, "models", "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    return None

def predict_tumor(image, model):
    img = image.convert("RGB")
    img_tensor = test_transforms(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0).numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_class, confidence, probs

def estimate_tumor_size(image, pred_class):
    if pred_class == "no_tumor":
        return None, None, None
    gray = image.convert("L")
    img_array = np.array(gray)
    threshold = np.percentile(img_array, 85)
    mask = img_array > threshold
    tumor_pixels = int(np.sum(mask))
    total_pixels = mask.size
    ratio = tumor_pixels / total_pixels
    estimated_size_mm = ratio * 50
    if ratio < 0.05:
        stage = "Early (Small)"
    elif ratio < 0.15:
        stage = "Moderate (Medium)"
    else:
        stage = "Advanced (Large)"
    return estimated_size_mm, ratio, stage

st.set_page_config(page_title="Brain Tumor Detection AI", layout="wide", page_icon="🧠")
st.title("🧠 Brain Tumor Detection & Grading AI")
st.markdown("---")

model = load_model()
if model is None:
    st.warning("No trained model found. Run `python src/brain_tumor_pipeline.py` first.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload MRI Scan")
    uploaded = st.file_uploader("Choose a brain MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded MRI", use_column_width=True)
    else:
        st.info("Upload an MRI image to analyze")
        # show sample from test set
        sample_dir = os.path.join(PROJECT_DIR, "data", "test")
        if os.path.exists(sample_dir):
            classes = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
            if classes:
                import random
                chosen_cls = random.choice(classes)
                cls_dir = os.path.join(sample_dir, chosen_cls)
                samples = [f for f in os.listdir(cls_dir) if f.endswith((".jpg", ".png"))]
                if samples:
                    sample_path = os.path.join(cls_dir, random.choice(samples))
                    image = Image.open(sample_path)
                    st.caption(f"Sample: {chosen_cls}")
                    st.image(image, use_column_width=True)

if uploaded or "image" in dir() or "sample_path" in dir():
    if uploaded:
        image = Image.open(uploaded)
    elif "image" in dir():
        pass
    else:
        image = Image.open(sample_path)

    pred_class, confidence, probs = predict_tumor(image, model)
    info = CLASS_LABELS[pred_class]
    emoji = CLASS_EMOJIS[pred_class]
    color = COLORS[pred_class]
    est_size, ratio, stage = estimate_tumor_size(image, pred_class)

    with col2:
        st.subheader("📋 Diagnosis Report")
        st.markdown(f"### {emoji} **{info['display']}**")
        if pred_class == "no_tumor":
            st.success(f"**No tumor detected** — Confidence: {confidence*100:.1f}%")
        else:
            st.error(f"**Tumor detected** — Confidence: {confidence*100:.1f}%")
            st.warning(f"**Tumor Grade:** {info['grade']}")
            st.info(f"**Risk Level:** {info['risk']}")

            if est_size:
                with st.container(border=True):
                    st.markdown("**📏 Size Estimation**")
                    st.metric("Estimated Size", f"{est_size:.1f} mm")
                    st.metric("Tumor Ratio", f"{ratio*100:.1f}%")
                    st.metric("Growth Stage", stage)
                    st.caption("Size estimated from pixel intensity — clinical validation required")

        st.markdown("---")
        st.markdown("**Confidence Scores**")
        prob_df = pd.DataFrame({
            "Class": [CLASS_LABELS[c]["display"] for c in CLASS_NAMES],
            "Confidence": probs * 100,
        }).sort_values("Confidence", ascending=False)
        fig = px.bar(prob_df, x="Class", y="Confidence", color="Class",
                     color_discrete_map=COLORS, text_auto=".1f",
                     title="Prediction Probabilities")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**{info['desc']}**")

    # side-by-side results
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("🧠 Detection", f"{confidence*100:.1f}% confidence", 
                  "Tumor" if pred_class != "no_tumor" else "No Tumor")
    with col4:
        st.metric("📊 Class", info["display"])
    with col5:
        if pred_class != "no_tumor":
            st.metric("⚠️ Grade", info["grade"], info["risk"])
        else:
            st.metric("✅ Status", "Healthy", "No concerns")

st.markdown("---")
st.caption("Brain Tumor Detection AI v1.0 | ResNet18 Transfer Learning | MRI Classification + Grading + Size Estimation")
