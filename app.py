import streamlit as st
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Set the title and header
st.title("🌺 Iris Species Classifier")
st.markdown("A simple Decision Tree model deployed with Streamlit.")

# --- Load the Model ---
@st.cache_resource
def load_model():
    # Use 'rb' for read binary mode
    try:
        with open('decision_tree_iris.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'decision_tree_iris.pkl' not found.")
        return None

model = load_model()

if model is not None:
    
    # --- Sidebar for User Input ---
    st.sidebar.header('Input Parameters')
    
    def user_input_features():
        # Input widgets for the four features (using the ranges from the Iris dataset)
        sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)
        
        data = {
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width
        }
        features = pd.DataFrame(data, index=[0])
        return features

    df_input = user_input_features()

    st.subheader('User Input Parameters')
    st.write(df_input)

    # --- Prediction ---
    prediction = model.predict(df_input)
    prediction_proba = model.predict_proba(df_input)

    st.subheader('Prediction')
    
    # Get the class name for display
    # We must replicate the class names from the original training
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    st.success(f"The predicted Iris species is: **{prediction[0]}**")

    st.subheader('Prediction Probability')
    proba_df = pd.DataFrame(prediction_proba, columns=class_names)
    st.write(proba_df)
