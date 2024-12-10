import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.linear_model import LinearRegression
# Load the model
model = LinearRegression()
@st.cache_resource
def load_model():
    with open('/Users/dattran/Downloads/22521142_Lab6/BaiTap12/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
model = load_model()


# App title and description
st.title("💎 Diamond Price Predictor 💎")
st.write("""
Welcome to the Diamond Price Predictor!  
Simply input the carat weight of a diamond, and we will predict its price for you.  
Enjoy a sparkly and interactive experience!
""")


# User input
carat_input = st.number_input(
    label="Enter the carat weight of the diamond:",
    min_value=0.01,
    max_value=5.00,
    value=1.00,
    step=0.01,
    format="%.2f"
)

# Predict price
if st.button("💰 Predict Price"):
    # Prepare input for the model
    test_input = np.array([[carat_input]])
    predicted_price = model.predict(test_input)
    predicted_price = predicted_price.item()
    # Display the result
    st.success(f"✨ The predicted price for a diamond of {carat_input} carats is **${predicted_price:,.2f}**!")

# Footer
st.markdown("""
---
💡 Created by Tran Manh Phuc From Quang Tri 💖  
""")
