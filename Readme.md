Here’s a **`README.md`** file that includes both the traffic vehicle classification and diamond price regression projects, with a focus on Streamlit app deployment.

---
# Trần Mạnh Phúc - 22521142

## 🚦 Traffic Vehicle Classification and 💎 Diamond Price Prediction

This repository contains two machine learning projects:

1. **Traffic Vehicle Classification**:
   - A deep learning model built using TensorFlow to classify traffic signs into **58 different categories**.
   - Includes a **Streamlit web app** for easy testing of traffic sign classification.

2. **Diamond Price Prediction**:
   - A regression model trained to predict the price of diamonds based on their carat weight.
   - Includes a **Streamlit web app** where users can input the carat weight of a diamond and get an estimated price.

---

## 🛠️ How to Run the Streamlit Applications

Both projects are implemented as Streamlit apps. Follow these steps to set up and run the apps:

### Step 1: Install Dependencies
Ensure all required libraries are installed. Use the provided `requirements.txt`:

pip install -r requirements.txt


This installs the necessary libraries such as TensorFlow, Streamlit, scikit-learn, and others.

---

### Step 2: Run the Streamlit Applications

#### Traffic Vehicle Classification

To launch the **Traffic Vehicle Classification** app, run the following command:

streamlit run /path/to/traffic_vehicle_demo.py


Replace `/path/to/traffic_vehicle_demo.py` with the actual path to the Python script.

#### Diamond Price Prediction

To launch the **Diamond Price Prediction** app, run the following command:

streamlit run /path/to/diamond_price_demo.py


Replace `/path/to/diamond_price_demo.py` with the actual path to the Python script.

---

### Step 3: Test the Applications

#### Traffic Vehicle Classification:
1. Open the Streamlit app in your browser (usually available at `http://localhost:8501`).
2. Upload an image of a traffic sign.
3. The app will display:
   - The **predicted class name**.
   - The **confidence score** for the prediction.

#### Diamond Price Prediction:
1. Open the Streamlit app in your browser.
2. Enter the carat weight of a diamond in the provided input box.
3. The app will display:
   - The **predicted price** of the diamond.

---

## 🚦 Traffic Vehicle Classification

This project uses a **deep learning model** (pretrained Xception) fine-tuned to classify **58 different traffic sign categories**. 

### Challenges:
- **Class Imbalance**: Some categories have very few samples.
- **Noisy Data**: Images with low resolution or artifacts.
- **Small Dataset**: Limited data size reduces test performance.

Despite these challenges, the model achieves reasonable results and can classify traffic signs via the Streamlit app.

---

## 💎 Diamond Price Prediction

This project uses a **linear regression model** to predict the price of diamonds based on their carat weight.

### Features:
- Simple input: Enter the carat weight.
- Quick prediction: Displays the estimated price instantly.
- Easy interface: Designed for seamless user interaction.

---

## 📚 Technologies Used

- **TensorFlow**: For traffic sign classification.
- **Scikit-learn**: For diamond price regression.
- **Streamlit**: For building interactive web applications.
- **Python**: For data processing and modeling.

---

## 📜 License

This project is licensed under the MIT License. Feel free to use and modify the code as needed.

