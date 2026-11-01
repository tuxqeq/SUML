"""Streamlit interface for the Iris predictor.

Run from the project root:
    streamlit run app/app.py
"""

import streamlit as st

from predict import predict

st.title("Iris Species Predictor")
st.write(
    "This app predicts the species of an iris flower "
    "(Setosa, Versicolor or Virginica) from four measurements. "
    "Enter the values in **centimeters**, then click **Predict**."
)

# One numeric input per feature, in the training order.
# Defaults are a typical Setosa flower.
sepal_length = st.number_input(
    "Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1
)
sepal_width = st.number_input(
    "Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1
)
petal_length = st.number_input(
    "Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1
)
petal_width = st.number_input(
    "Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1
)

if st.button("Predict", type="primary"):
    # Same order as in training: sepal length, sepal width, petal length, petal width.
    features = [sepal_length, sepal_width, petal_length, petal_width]
    species = predict(features)
    st.success(f"Predicted species: **{species}**")
