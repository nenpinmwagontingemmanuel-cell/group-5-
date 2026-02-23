
import streamlit as st
import joblib
import pandas as pd

# Load the trained model and label encoder
model = joblib.load('random_forest_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Load the StandardScaler fitted on the continuous features
# To ensure consistency in scaling new inputs, we'll load the original dataframe
# and refit the scaler on the 'continuous_features' from the original df.
# This is a robust way to ensure that the scaler used for prediction
# matches the one used during training, especially if the scaler itself wasn't saved.
# For a more robust deployment, one would save the scaler as well.
# For this example, we'll simulate it by loading the original data to re-initialize the scaler.
# In a real-world scenario, you would save the 'scaler' object directly.
# Since `df` is available from the notebook's kernel state, we know the original continuous features.
# However, the streamlit app runs independently, so we'll need to re-create the scaler from scratch
# using some proxy min/max or by saving the scaler explicitly during training.
# For simplicity, let's assume the previous `StandardScaler` was used for `continuous_features`.
# We need the `mean_` and `scale_` (std dev) attributes from the *trained* scaler.
# If `scaler` object was saved: scaler = joblib.load('scaler.joblib')

# Assuming `scaler` object was saved along with the model and label_encoder during training
# However, it wasn't explicitly saved. So, for the app, we need to apply scaling logic.
# A simple way for a deployed app without the saved scaler is to use the `df.describe()`
# values from the training phase to manually scale, or ideally, save the scaler itself.
# For now, let's mock it with sample values or assume features are already scaled if `model` was trained on scaled data.
# Since the previous steps used `StandardScaler` on `precipitation`, `temp_max`, `temp_min`, `wind`,
# we need to ensure the same scaling is applied to user inputs.
# The most robust way is to save the scaler object itself along with the model.
# Let's assume we saved it. If not, this would be a point for improvement.
# For the purpose of this task, we will load the original data to create a new StandardScaler
# to mimic the previous scaling. This is NOT ideal for production but serves the current purpose
# given previous steps didn't save the scaler.

# NOTE: This part is a workaround. In a production scenario, you MUST save the `StandardScaler` object
# during training and load it here to ensure correct scaling of new inputs.
original_df = pd.read_csv('/content/seattle-weather.csv')
continuous_features_list = ['precipitation', 'temp_max', 'temp_min', 'wind']
from sklearn.preprocessing import StandardScaler
mock_scaler = StandardScaler()
mock_scaler.fit(original_df[continuous_features_list]) # Fit on original data to get scaling parameters

st.title('Weather Prediction App')
st.write("A simple app to predict weather based on meteorological features.")

st.write("### Enter Weather Parameters for Prediction")

# User input fields
precipitation = st.slider('Precipitation (mm)', 0.0, 60.0, 5.0, help="Amount of precipitation in mm.")
temp_max = st.slider('Max Temperature (°C)', -10.0, 40.0, 15.0, help="Maximum temperature in Celsius.")
temp_min = st.slider('Min Temperature (°C)', -15.0, 25.0, 5.0, help="Minimum temperature in Celsius.")
wind = st.slider('Wind Speed (m/s)', 0.0, 10.0, 2.0, help="Wind speed in m/s.")
month = st.selectbox('Month', range(1, 13), index=0, help="Month of the year (1 for January, 12 for December).")

# Prediction button
if st.button('Predict Weather'):
    # Create a DataFrame for prediction, ensuring column order matches training data
    input_data = pd.DataFrame([{
        'precipitation': precipitation,
        'temp_max': temp_max,
        'temp_min': temp_min,
        'wind': wind,
        'month': month
    }])

    # Scale the continuous features using the mock_scaler
    input_data[continuous_features_list] = mock_scaler.transform(input_data[continuous_features_list])

    # Make prediction
    prediction_encoded = model.predict(input_data)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    st.write("--- ")
    st.write("### Prediction Results")
    st.success(f"The predicted weather is: **{prediction_label[0].capitalize()}**")

