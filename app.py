import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import os

# Check if model and scaler exist
MODEL_PATH = "svm_digit_model.joblib"
SCALER_PATH = "digit_scaler.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or scaler file not found. Please ensure 'svm_digit_model.joblib' and 'digit_scaler.joblib' exist.")
    st.stop()

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Streamlit UI
st.title("🔢 Handwritten Digit Classifier")
st.write("Upload a **28x28 or 8x8 grayscale image** of a digit (0–9) to classify.")

uploaded_file = st.file_uploader("Choose a digit image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image to match the 8x8 format of the dataset
    image_resized = image.resize((8, 8), Image.Resampling.LANCZOS)
    image_inverted = ImageOps.invert(image_resized)
    image_array = np.array(image_inverted).astype(float)

    # Scale image data like the dataset: pixel values between 0–16
    image_scaled = (image_array / 255.0) * 16.0
    image_flatten = image_scaled.flatten().reshape(1, -1)

    # Apply the saved scaler
    image_flatten_scaled = scaler.transform(image_flatten)

    # Make prediction
    prediction = model.predict(image_flatten_scaled)[0]
    st.success(f"✅ Predicted Digit: **{prediction}**")
