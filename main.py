import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load your model
model_loaded = load('Gradient_boosting_model.joblib')

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
    
    # Variable to store selected columns for upload option
    selected_columns = []

    if option == "Enter values":
        # Dynamically create input boxes for each feature
        user_input = {}
        for variable in variable_list:
            value = st.number_input(f"Enter value for {variable}:", min_value=0.0, format="%.4f")
            if value < 0:
                st.error(f"{variable} cannot be a negative value.")
            user_input[variable] = value
        
        # Predict button
        if st.button('Predict'):
            # Convert the input into a DataFrame
            input_df = pd.DataFrame([user_input])
            predict_and_display(input_df)  # Single prediction based on user input
    
    else:  # Option to upload file
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
        if uploaded_file is not None:
            try:
                # Load the first sheet, assuming headers are in the second row (index 1)
                data = pd.read_excel(uploaded_file, header=1)
                
                # Print column names for debugging
                st.write("Columns in uploaded file:", data.columns.tolist())
                
                # Select columns
                selected_columns = st.multiselect("Select columns for prediction:", data.columns.tolist(), default=variable_list)
                
                # Ensure that selected columns match variable_list for future input use
                if set(selected_columns).issubset(data.columns):
                    data = data[selected_columns]  # Data based on selected columns
                    predict_and_display(data)  # File-based prediction
                    
                    # Populate the enter values form with selected columns for further testing
                    st.sidebar.write("Play with selected columns:")
                    user_input = {}
                    for variable in selected_columns:
                        user_input[variable] = st.sidebar.number_input(f"Enter value for {variable}:", min_value=0.0, format="%.4f")
                    
                    if st.sidebar.button("Test new values"):
                        input_df = pd.DataFrame([user_input])
                        predict_and_display(input_df)
                else:
                    st.error("The uploaded file does not contain all the selected columns.")
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")

def predict_and_display(data):
    try:
        # Make predictions
        predictions = model_loaded.predict(data)
        
        # Combine inputs and predictions into a DataFrame
        results_df = data.copy()
        results_df['Prediction'] = predictions
        
        # Display the results in a scrollable dataframe
        st.write("Prediction Results:")
        st.dataframe(results_df, use_container_width=True)  # Scrollable table
        
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
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
