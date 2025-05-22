import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib

# Load trained SVM model and scaler
model = joblib.load("svm_digit_model.joblib")
scaler = joblib.load("digit_scaler.joblib")

st.title("Digit Classifier (8x8 SVM)")
st.write("Upload a **28x28 or higher** grayscale image of a digit (0-9).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    # Convert to grayscale and invert if needed
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((8, 8))  # Resize to 8x8
    image_np = np.array(image)
    image_np = image_np / 16.0  # Normalize to same scale as sklearn's digits
    return image_np.reshape(1, -1)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False)

    if st.button("Predict"):
        input_data = preprocess_image(image)
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        st.success(f"Predicted Digit: {prediction[0]}")
