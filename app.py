import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the TensorFlow model
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    st.error(f'Error loading model: {e}')
    raise

# Function to preprocess the image and make predictions
def predict(image, model):
    image = image.resize((128, 128))  # Resize to the size your model expects
    image = np.array(image) / 255.0  # Normalize the image
    image = image.reshape((1, 128, 128, 3))  # Reshape to add batch dimension
    prediction = model.predict(image)
    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    return class_names[np.argmax(prediction)]

st.title('Rice Image Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make prediction
        predicted_class = predict(image, model)
        st.write(f'Predicted class: {predicted_class}')
    except Exception as e:
        st.error(f'Error during prediction: {e}')
