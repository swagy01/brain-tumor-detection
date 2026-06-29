import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import gdown
import os

st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_best_model():
    model_path = "final_model.h5"
    if not os.path.exists(model_path):
        st.info("⏳ Downloading model... please wait (first time only)")
        gdown.download(
            "https://drive.google.com/uc?id=17xaIgEqK9T0OazSDMh5OREc1V0TbKZbW",
            model_path, quiet=False
        )
    model = load_model(model_path)
    return model

model = load_best_model()

def get_gradcam(model, img_array, last_conv_layer="block5_conv3"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(
        (img * 255).astype(np.uint8), 0.6,
        heatmap_colored, 0.4, 0
    )
    return superimposed

# ── UI ─────────────────────────────────────────────────
st.title("🧠 Brain Tumor Detection System")
st.markdown("**Upload a Brain MRI Image to detect if a tumor is present.**")
st.markdown("---")
st.info("**Model:** VGG16 Fine-tuned | **Test Accuracy:** 100% | **Dataset:** 351 Brain MRI Images")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI scan image"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    pred_prob = model.predict(img_input, verbose=0)[0][0]

    if pred_prob < 0.5:
        result = "TUMOR DETECTED"
        confidence = (1 - pred_prob) * 100
        emoji = "🔴"
    else:
        result = "NO TUMOR DETECTED"
        confidence = pred_prob * 100
        emoji = "🟢"

    heatmap = get_gradcam(model, img_input.astype("float32"))
    overlay = overlay_gradcam(img_normalized, heatmap)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original MRI")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        fig, ax = plt.subplots()
        ax.imshow(heatmap, cmap="jet")
        ax.axis("off")
        st.pyplot(fig)

    with col3:
        st.subheader("AI Focus Overlay")
        st.image(overlay, use_column_width=True)

    st.markdown("---")

    if pred_prob < 0.5:
        st.error(f"{emoji} **{result}**\nConfidence: **{confidence:.2f}%**")
    else:
        st.success(f"{emoji} **{result}**\nConfidence: **{confidence:.2f}%**")

    st.markdown("---")
    st.warning("⚠️ Disclaimer: This is an AI-based system for educational purposes only. Always consult a qualified medical professional for diagnosis.")

else:
    st.markdown("### How to use:")
    st.markdown("1. Click **Browse files** above")
    st.markdown("2. Upload any **Brain MRI scan** image (.jpg or .png)")
    st.markdown("3. AI will instantly predict **Tumor / No Tumor**")
    st.markdown("4. **Grad-CAM heatmap** will show which area AI focused on")
    st.markdown("---")
    st.markdown("### Project Summary:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", "VGG16 FT")
    col2.metric("Accuracy", "100%")
    col3.metric("Dataset", "351 MRIs")
    col4.metric("Models Tested", "5")
