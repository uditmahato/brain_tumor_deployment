import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # Replace with your model's library
import torch             # Use PyTorch if applicable

# Set Streamlit page configuration
st.set_page_config(page_title="MRI Tumor Classification", layout="centered")

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

# Class labels
class_labels = {
    0: "Glioma Tumor",
    1: "No Tumor",
    2: "Meningioma Tumor",
    3: "Pituitary Tumor"
}

# Streamlit UI
st.title("MRI Tumor Classification")
st.markdown("""
### About the App
This application uses a machine learning model to classify MRI brain scans into one of the following categories:
- **Glioma Tumor**
- **No Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**

Upload an MRI scan image, and the model will analyze and predict the tumor type.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI scan image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess and predict
    with st.spinner("Analyzing the image..."):
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        print(f"Raw Predictions: {prediction}")  # Print raw probabilities
        print(f"Predicted Class: {np.argmax(prediction)}")  # Print predicted class index
        result = np.argmax(prediction, axis=1)[0]  # Get the class label

    # Debugging: Display raw predictions
    st.markdown(f"### Raw Model Predictions:\n{prediction[0]}")

    # Display the result
    st.success(f"Prediction: **{class_labels[result]}**")

    # Print the predicted label to the terminal
    print(f"Predicted Label: {class_labels[result]}")

else:
    st.info("Please upload an MRI scan image to start the classification.")
