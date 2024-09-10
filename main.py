import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# Define the model path
MODEL_PATH = 'Gradient_boosting_model.joblib'

# Streamlit app starts here
def main():
    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        return

    try:
        # Load your model
        model_loaded = load(MODEL_PATH)
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return

    # List of variables for prediction
    variable_list = [
        'aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine', 
        'chromium', 'copper', 'flouride', 'bacteria', 'viruses', 'lead', 
        'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium', 
        'selenium', 'silver', 'uranium'
    ]

    # Title of your app
    st.title("Water Safety Prediction App")
    
    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose how to input data", ["Enter values", "Upload file"])
    
    # Variable to store selected row from file upload
    selected_row_data = None

    if option == "Enter values":
        # Check if there's a row data preloaded from file upload
        if selected_row_data is not None:
            st.write("Populated from file upload:")
            st.write(selected_row_data)

        # Dynamically create input boxes for each feature
        user_input = {}
        for variable in variable_list:
            default_value = selected_row_data.get(variable, 0.0) if selected_row_data else 0.0
            user_input[variable] = st.number_input(f"Enter value for {variable}:", min_value=0.0, value=default_value, format="%.4f")
        
        # Predict button
        if st.button('Predict'):
            # Convert the input into a DataFrame
            input_df = pd.DataFrame([user_input])
            predict_and_display(input_df, model_loaded)  # Single prediction based on user input
    
    else:  # Option to upload file
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
        if uploaded_file is not None:
            try:
                # Load the first sheet, assuming headers are in the second row (index 1)
                data = pd.read_excel(uploaded_file, header=1)
                
                # Print the dataframe for user to select a row
                st.write("Uploaded data preview:")
                st.dataframe(data)
                
                # Let user select a row by index
                row_index = st.number_input("Select row index for input:", min_value=0, max_value=len(data)-1, step=1)
                
                # If a valid row is selected, show the selected row data
                if row_index is not None and 0 <= row_index < len(data):
                    st.write(f"Selected row {row_index}:")
                    selected_row_data = data.iloc[row_index].to_dict()
                    st.write(selected_row_data)
                
                    # Allow the user to play with the values in "Enter values" section
                    st.sidebar.write("Play with selected row values:")
                    user_input = {}
                    for variable in variable_list:
                        value = selected_row_data.get(variable, 0.0)
                        user_input[variable] = st.sidebar.number_input(f"Enter value for {variable}:", min_value=0.0, value=value, format="%.4f")
                    
                    if st.sidebar.button("Test new values"):
                        input_df = pd.DataFrame([user_input])
                        predict_and_display(input_df, model_loaded)
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")

def predict_and_display(data, model):
    try:
        # Make predictions
        predictions = model.predict(data)
        
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
