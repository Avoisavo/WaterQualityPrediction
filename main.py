import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load your model and any other necessary components
model_loaded = load('Gradient_boosting_model.joblib')  # Assuming you have a trained model for this
scaler_loaded = load('scaler.joblib')  # Load any scaler if used during model training

# List of variables for prediction
variable_list = [
    'aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine', 
    'chromium', 'copper', 'flouride', 'bacteria', 'viruses', 'lead', 
    'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium', 
    'selenium', 'silver', 'uranium'
]

# Streamlit app starts here
def main():
    # Title of your app
    st.title("Water Safety Prediction App")
    
    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose how to input data", ["Enter values", "Upload file"])
    
    if option == "Enter values":
        # Dynamically create input boxes for each feature
        user_input = {}
        for variable in variable_list:
            user_input[variable] = st.number_input(f"Enter value for {variable}:", min_value=0.0, format="%.4f")
        
        # Predict button
        if st.button('Predict'):
            # Convert the input into a DataFrame
            input_df = pd.DataFrame([user_input])
            predict_and_display(input_df)  # Single prediction based on user input
    
    else:  # Option to upload file
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            # Check if the file contains all necessary columns
            if set(variable_list).issubset(data.columns):
                predict_and_display(data)  # File-based prediction
            else:
                st.error("The uploaded file does not contain all the required columns.")
    
def predict_and_display(data):
    # If you use scaling, apply the scaler here
    scaled_data = scaler_loaded.transform(data)
    
    # Make predictions
    predictions = model_loaded.predict(scaled_data)
    
    # Combine inputs and predictions into a DataFrame
    results_df = data.copy()
    results_df['Prediction'] = predictions
    
    # Display the results in a table
    st.write("Prediction Results:")
    st.table(results_df)
    
    # Display a histogram of the predictions
    st.write("Histogram of Predictions:")
    fig, ax = plt.subplots()
    prediction_counts = pd.Series(predictions).value_counts().sort_index()
    prediction_counts.plot(kind='bar', ax=ax)
    ax.set_title("Number of Safe and Unsafe Predictions")
    ax.set_xlabel("Category (0=Not Safe, 1=Safe)")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure y-axis has integer ticks
    st.pyplot(fig)

if __name__ == '__main__':
    main()
