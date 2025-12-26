# 🧠 Breast Cancer Prediction Using Neural Network

This project is a **Machine Learning web application** that predicts whether a breast tumor is **Benign or Malignant** using a **Neural Network model**.
The app is built with **TensorFlow/Keras** and deployed using **Streamlit**.

---

## 🚀 Live Demo

👉 **Streamlit App:**
** https://breastcancerpredictionusingnn-ea5hqfhppah5qdjvbtw24t.streamlit.app/*

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early detection can significantly improve treatment outcomes.
This application helps in predicting breast cancer based on medical input features using a trained **Neural Network classifier**.

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Machine Learning:** TensorFlow, Keras
* **Data Processing:** NumPy, Pandas, Scikit-learn
* **Model Deployment:** Streamlit
* **Model Persistence:** Joblib
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
breast_cancer_prediction_using_nn/
│
├── app.py                   # Streamlit application
├── breast_cancer_model.keras # Trained neural network model
├── scaler.pkl               # StandardScaler used for input normalization
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
```

---

## ⚙️ How It Works

1. User enters medical feature values through the UI.
2. Input data is **scaled using StandardScaler**.
3. Scaled data is passed to the **Neural Network model**.
4. Model predicts:

   * **Benign**
   * **Malignant**
5. Result is displayed instantly on the web interface.

---

## 🧪 Model Details

* **Algorithm:** Neural Network (Feed-Forward)
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Evaluation Metric:** Accuracy

---

## ▶️ How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/breast_cancer_prediction_using_nn.git
cd breast_cancer_prediction_using_nn
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📦 Requirements

```txt
streamlit>=1.52.0
tensorflow>=2.20.0
numpy>=1.26.0
scikit-learn>=1.7.0
joblib
```

## 👩‍💻 Author

**Vaidehi Vilas Bankar**
🔗 GitHub: *https://github.com/vaidehibankar21*
🔗 LinkedIn: *www.linkedin.com/in/vaidehi-bankar-167524295*
