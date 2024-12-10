
# 🚦 Traffic Vehicle Classification Dataset and Demo

## 📂 Dataset Overview

This project is built on a **traffic vehicle dataset** consisting of **58 different classes**, representing various traffic signs and rules. Examples include speed limits, no-entry signs, stop signs, and more. However, the dataset poses significant challenges:

- **Class Imbalance**: Some classes have very few samples compared to others, leading to potential generalization issues.
- **Noisy Data**: Many images contain distortions, low resolution, or artifacts, making it hard for models to learn effectively.
- **Small Dataset Size**: The dataset is relatively small, which limits the performance achievable on the test set.

Despite these challenges, a pretrained **Xception model** was fine-tuned on this dataset to achieve reasonable performance.

---

## 🚀 Web Demo Instructions

Follow the steps below to run the web demo and classify traffic signs using the trained model.

### 🛠️ Step 1: Install Requirements
Ensure all necessary dependencies are installed by running the following command:

```bash
pip install -r requirements.txt

🖥️ Step 2: Run the Streamlit Application
streamlit run /path/to/Web_Demo.py 
Replace /path/to/Web_Demo.py with the actual path to the Web_Demo.py script. 

streamlit run Web_Demo.py

📸 Step 3: Classify Traffic Sign Images
Open the web app in your browser (the URL is typically http://localhost:8501).
Upload an image from the test dataset using the "Choose an image file" button.
View the model's prediction, including:
The predicted class name.
The confidence score for the prediction.

💡 Notes and Observations
The model performs well under certain conditions but may misclassify due to the imbalanced and noisy dataset.
Improvements such as data augmentation and additional training data could enhance the model's accuracy.
The demo provides a straightforward interface for testing individual traffic sign images.