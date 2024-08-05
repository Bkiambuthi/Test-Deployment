import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Load the pickled model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to preprocess the image and make predictions
def predict(image, model):
    # Preprocess the image to the format the model expects
    image = image.resize((128, 128))  # Adjust size as needed
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1] range if necessary
    image = image.reshape((1, 128, 128, 3))  # Reshape to (1, 128, 128, 3)
    
    # Make a prediction
    prediction = model.predict(image)
    return prediction

# Streamlit app
st.title("Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        prediction = predict(image, model)
        st.write(f'Prediction: {prediction[0]}')
    except Exception as e:
        st.write(f"Error: {e}")
