import streamlit as st
import pandas as pd
# Notice how we import from our local 'src' folder
from src.data_loader import load_sales_data
from src.model import SimpleForecaster

# 1. App Configuration
st.set_page_config(page_title="AI Sales Forecaster", layout="wide")
st.title("📊 AI Engineering: Sales Forecaster")

# 2. Sidebar Controls
days_to_train = st.sidebar.slider("Days of History", 50, 365, 100)
days_to_predict = st.sidebar.slider("Days to Forecast", 7, 90, 30)

# 3. Execution Logic
if st.button("Generate Forecast"):
    # Step A: Load Data
    with st.spinner("Loading Data..."):
        df = load_sales_data(days=days_to_train)
    
    # Step B: Initialize and Train Model
    # This is where we use the Class we created
    forecaster = SimpleForecaster()
    forecaster.train(df)
    
    # Step C: Predict
    forecast_df = forecaster.predict_next_days(df, days_to_predict=days_to_predict)
    
    # Step D: Visualize
    st.subheader("Historical Data vs Forecast")
    
    # Combine data for plotting
    chart_data = pd.concat([
        df.set_index('ds').rename(columns={'y': 'Actual Sales'}),
        forecast_df.set_index('ds').rename(columns={'y_pred': 'Predicted Sales'})
    ], axis=1)
    
    st.line_chart(chart_data)
    
    st.success("Forecast generated successfully!")