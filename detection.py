import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # Replace with your model's library
import torch             # Use PyTorch if applicable

# Load your model
@st.cache_resource
def load_model():
    # Example for TensorFlow
    return tf.keras.models.load_model("my_model.h5")
    # Example for PyTorch
    # return torch.load("path_to_your_model.pth")

model = load_model()

# Preprocess image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize based on your model's input size
    image_array = np.array(image) / 255.0  # Normalize (adjust as required)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit UI
st.title("MRI Scan Classification")
st.write("Upload an MRI scan image to classify.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    with st.spinner("Processing..."):
        preprocessed_image = preprocess_image(image)
       
        # Example prediction for TensorFlow
        prediction = model.predict(preprocessed_image)
        result = np.argmax(prediction, axis=1)  # Get the class label

       
    st.success(f"Prediction: Class {"positive" if result == 1 else "Negative"}")

