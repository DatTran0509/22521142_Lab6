import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Define the model architecture
@st.cache_resource
def load_model():
    input_shape = (299, 299, 3)  # Update this based on your model
    pretrained_base = Xception(include_top=False, input_shape=input_shape, pooling='avg', weights="imagenet")
    pretrained_base.trainable = False

    model = Sequential([
        pretrained_base,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.1),
        Dense(58, activation='softmax')  # Replace 58 with the number of classes in your dataset
    ])
    # Load the saved best weights
    model.load_weights('/Users/dattran/Downloads/22521142_Lab6/BaiTap11/best_weights.h5')  # Update with your path
    return model

model = load_model()

# Define the class names
class_names = [
    "Speed limit (5km/h)", "Speed limit (15km/h)", "Speed limit (30km/h)", "Speed limit (40km/h)", 
    "Speed limit (50km/h)", "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", 
    "Dont Go straight or left", "Dont Go straight or Right", "Dont Go straight", "Dont Go Left", 
    "Dont Go Left or Right", "Dont Go Right", "Dont overtake from Left", "No Uturn", 
    "No Car", "No horn", "Speed limit (40km/h)", "Speed limit (50km/h)", 
    "Go straight or right", "Go straight", "Go Left", "Go Left or right", 
    "Go Right", "keep Left", "keep Right", "Roundabout mandatory", "watch out for cars", "Horn", 
    "Bicycles crossing", "Uturn", "Road Divider", "Traffic signals", "Danger Ahead", "Zebra Crossing", 
    "Bicycles crossing", "Children crossing", "Dangerous curve to the left", "Dangerous curve to the right", 
    "Unknown1", "Unknown2", "Unknown3", "Go right or straight", "Go left or straight", "Unknown4", 
    "ZigZag Curve", "Train Crossing", "Under Construction", "Unknown5", "Fences", 
    "Heavy Vehicle Accidents", "Unknown6", "Give Way", "No stopping", "No entry", 
    "Unknown7", "Unknown8"
]

def preprocess_image(image):
    """Preprocess the uploaded image to fit the model input requirements."""
    image = image.resize((299, 299))  # Xception requires 299x299 input size
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    if image.shape[-1] == 4:  # If the image has an alpha channel, remove it
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

st.title("Traffic Sign Classification with Pretrained Model")
st.write("Upload an image to classify the traffic sign.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    st.write("Processing the image...")
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class_idx = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]

    # Display the results
    st.write(f"**Predicted Class:** {predicted_class_name}")
    st.write(f"**Confidence:** {confidence:.2f}")
