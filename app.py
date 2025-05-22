import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import joblib
import os

# Load model and scaler
model_path = "svm_digit_model.joblib"
scaler_path = "digit_scaler.joblib"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("❌ Model or scaler file not found. Please ensure 'svm_digit_model.joblib' and 'digit_scaler.joblib' exist.")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# App title
st.title("🧠 Handwritten Digit Classifier (SVM)")

# Upload image
uploaded_file = st.file_uploader("Upload an 8x8 grayscale digit image (64x64, will be resized)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    image_resized = ImageOps.fit(image, (8, 8), method=Image.LANCZOS)
    
    # Optional: show image
    st.image(image_resized.resize((128, 128)), caption="Uploaded Image (resized to 8x8)", use_column_width=False)

    # Normalize image to match sklearn format (0-16 pixel range)
    image_np = np.asarray(image_resized)
    image_np = np.clip((16 - (image_np / 16)), 0, 16)  # invert and scale
    image_flat = image_np.flatten().reshape(1, -1)

    # Standardize using saved scaler
    image_scaled = scaler.transform(image_flat)

    # Predict
    prediction = model.predict(image_scaled)[0]

    st.success(f"✅ Predicted Digit: **{prediction}**")

    # Show raw values as heatmap
    st.write("Image Data (8x8):")
    fig, ax = plt.subplots()
    ax.imshow(image_np, cmap='gray', interpolation='nearest')
    ax.set_title(f'Predicted: {prediction}')
    ax.axis('off')
    st.pyplot(fig)
