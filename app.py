import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_digit_model.h5")

model = load_model()

# Preprocess uploaded image
def preprocess_image(image):
    img = image.convert('L')           # Convert to grayscale
    img = img.resize((28, 28))         # Resize to 28x28
    img_array = np.array(img)
    img_array = 255 - img_array        # Invert image (white digit on black bg)
    img_array = img_array / 255.0      # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# Streamlit UI
st.title("Handwritten Digit Recognizer (MNIST Model)")

uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)

        st.success(f"Predicted Digit: **{predicted_digit}**")
        st.bar_chart(prediction[0])
