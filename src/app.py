from pickle import load
import streamlit as st
import numpy as np

# Load the model
model = load(open("../models/decision_tree_default_42.pkl", 'rb'))

# Define the dict:
class_dict = {
    "0" : "Bad",
    "1" : "Good"
}

# Set the title
st.title("Red Wine Quality Predictor")


# Get user inputs from the form
fixed_acidity = st.number_input("fixed_acidity")
volatile_acidity = st.number_input("volatile_acidity")
citric_acid = st.number_input("citric_acid")
residual_sugar = st.number_input("residual_sugar")
chlorides = st.number_input("chlorides")
free_sulfur_dioxide = st.number_input("free_sulfur_dioxide")
total_sulfur_dioxide = st.number_input("total_sulfur_dioxide")
density = st.number_input("density")
pH = st.number_input("pH")
sulphates = st.number_input("sulphates")
alcohol = st.number_input("alcohol")

# Create a feature vector with user inputs
feature_vector = (fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide,density, pH, sulphates, alcohol)

if st.button("Evaluate"):
    prediction = model.predict([feature_vector])
    pred_class = class_dict[str(prediction[0])]  # Convert prediction to a string before using it as a key in class_dict
    st.write("The quality of this red wine is:", pred_class)