import streamlit as st
import numpy as np
import pickle
import os

# Load the pre-trained model
MODEL_PATH = 'models/model.pkl'  # Update this path if necessary

# Check if the model file exists
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    st.error(f"Model file not found at {MODEL_PATH}. Please check the file path.")
    st.stop()  # Stop further execution if the model file is not found

# Streamlit app title
st.title("Breast Cancer Prediction")

# Instructions
st.write("Enter the features separated by commas (e.g., `2.3, 4.1, 1.8, ...`)")

# Input for features
features_input = st.text_input("Enter Features:")

# Prediction button
if st.button("Predict"):
    try:
        # Convert input to numpy array
        features = [float(x) for x in features_input.split(',')]
        np_features = np.asarray(features, dtype=np.float32)

        # Prediction
        pred = model.predict(np_features.reshape(1, -1))
        message = "Cancerous" if pred[0] == 1 else "Not Cancerous"

        # Display the prediction result
        st.success(f"Predicted Result: {message}")
    except ValueError:
        st.error("Please enter valid numerical values, separated by commas.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
