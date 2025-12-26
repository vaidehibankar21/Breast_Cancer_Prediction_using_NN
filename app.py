import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
from datetime import datetime

# ---------------------------
# Cached Model and Scaler Loading (TFLite)
# ---------------------------
@st.cache_resource
def load_model_and_scaler():
    interpreter = tf.lite.Interpreter(model_path="breast_cancer_model.tflite")
    interpreter.allocate_tensors()

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return interpreter, scaler

model, scaler = load_model_and_scaler()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide"
)

# ---------------------------
# Header Image
# ---------------------------
header_img = Image.open("CGImg.png")
st.image(header_img, width=700)
st.title("🎗️ Breast Cancer Prediction App")
st.write("Enter the 30 tumor features below to predict whether the tumor is benign or malignant.")

# ---------------------------
# Viewer Tracking File
# ---------------------------
VIEWERS_FILE = "viewers.csv"
if not os.path.exists(VIEWERS_FILE):
    pd.DataFrame(columns=["Name", "Rating", "Date", "Liked"]).to_csv(VIEWERS_FILE, index=False)

viewers_df = pd.read_csv(VIEWERS_FILE)
viewers_df["Date"] = pd.to_datetime(viewers_df["Date"], errors="coerce")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Input Features")
feature_names = [
    "radius_mean","texture_mean","perimeter_mean","area_mean",
    "smoothness_mean","compactness_mean","concavity_mean",
    "concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se",
    "smoothness_se","compactness_se","concavity_se",
    "concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst",
    "smoothness_worst","compactness_worst","concavity_worst",
    "concave points_worst","symmetry_worst","fractal_dimension_worst"
]

inputs = [st.sidebar.number_input(name, value=0.0) for name in feature_names]

# ---------------------------
# Feedback & Rating
# ---------------------------
st.sidebar.header("Feedback & Rating")
st.session_state.user_name = st.text_input(
    "Your Name",
    st.session_state.get("user_name", "Anonymous")
)
st.session_state.user_rating = st.slider(
    "Rate this App",
    1, 5,
    st.session_state.get("user_rating", 5)
)

if st.sidebar.button("Submit Feedback"):
    today = pd.Timestamp(datetime.today().date())
    already_rated = viewers_df[
        (viewers_df["Name"] == st.session_state.user_name) &
        (viewers_df["Date"] == today)
    ]

    if already_rated.empty:
        new_row = pd.DataFrame({
            "Name": [st.session_state.user_name],
            "Rating": [st.session_state.user_rating],
            "Date": [today],
            "Liked": [0]
        })
        viewers_df = pd.concat([viewers_df, new_row], ignore_index=True)
        viewers_df.to_csv(VIEWERS_FILE, index=False)
        st.sidebar.success("Thank you! Your rating has been recorded.")
    else:
        st.sidebar.info("You have already rated today.")

# ---------------------------
# Display Viewer Stats
# ---------------------------
st.sidebar.subheader("Viewer Stats")
st.sidebar.markdown(f"**Total Views:** {len(viewers_df)}")
st.sidebar.markdown(f"**Average Rating:** {viewers_df['Rating'].mean():.1f} ⭐")
st.sidebar.markdown("**Viewers:**")
for name in viewers_df["Name"]:
    st.sidebar.markdown(f"- {name}")

# ---------------------------
# Prediction (TFLite)
# ---------------------------
if st.button("Predict"):
    scaled_input = scaler.transform([inputs]).astype("float32")

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]["index"], scaled_input)
    model.invoke()
    prob = model.get_tensor(output_details[0]["index"])[0][0]

    result = "Malignant" if prob > 0.5 else "Benign"
    color = "red" if result == "Malignant" else "green"

    st.markdown(
        f"<h3 style='color:{color}'>{result} (Probability: {prob:.3f})</h3>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.write("Thank you for using the Breast Cancer Prediction App!")