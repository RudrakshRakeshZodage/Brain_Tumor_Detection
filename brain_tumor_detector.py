import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="🧠 Brain Tumor Detector", layout="centered")

# Load your model (ensure this file is present in the same folder)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.h5")
    return model

model = load_model()

# Image Preprocessing Function (resize + normalize)
def preprocess_image(image):
    image = image.resize((150, 150))  # Adjust as per model input size
    image = np.array(image)
    image = image / 255.0  # Normalize
    image = image.reshape(1, 150, 150, 3)
    return image

# Streamlit UI
st.title("🧠 Brain Tumor Detection from MRI Scans")
st.markdown("""
Upload an MRI scan image, and the system will detect whether a **brain tumor** is present or not.  
""")

uploaded_file = st.file_uploader("📤 Upload MRI Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="🧾 Uploaded MRI Scan", use_column_width=True)
    
    with st.spinner("🔍 Analyzing MRI..."):
        image = Image.open(uploaded_file).convert("RGB")
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]

        if prediction > 0.5:
            st.error("🚨 Tumor Detected!", icon="🚑")
        else:
            st.success("✅ No Tumor Detected", icon="🧬")

        st.write(f"🧠 Confidence Score: `{prediction:.3f}`")

# Optional Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>© 2025 Rudraksh Zodage. All rights reserved.</div>",
    unsafe_allow_html=True
)

