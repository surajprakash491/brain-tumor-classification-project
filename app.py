import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('brain_tumor_classifier_model.h5')

# Define the labels
labels = ['glioma','meningioma','notumor','pituitary']

st.title('Brain Tumor Classification')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded image to a format suitable for the model
    img = Image.open(uploaded_file).convert("RGB")  # Ensure 3 channels
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image to match input shape
    img = img.resize((150,150))  # Ensure same size as used in model
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Display the result
    st.write(f"Predicted Class: {labels[predicted_class]}")
