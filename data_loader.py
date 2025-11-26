import pandas as pd

# This URL is a raw hosted version of the Superstore dataset
DATA_URL = "https://raw.githubusercontent.com/curran/data/gh-pages/superstoreSales/superstoreSales.csv"

def load_sales_data(days=None):
    """
    Fetches real Superstore data from a remote URL.
    Aggregates transactional data into daily sales sums for forecasting.
    """
    try:
        # 1. Load Data from URL (Engineering: Automated Ingestion)
        # encoding='latin1' is often needed for this specific dataset
        df = pd.read_csv(DATA_URL, encoding='latin1')
        
        # 2. Data Cleaning & formatting
        # We need to ensure the date column is actually a datetime object
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        
        # 3. Aggregation (Engineering: Feature Engineering)
        # The raw data is by 'Transaction'. We need it by 'Day' for forecasting.
        daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
        
        # 4. Standardizing for our Model
        # Our model expects columns named 'ds' (date) and 'y' (value)
        daily_sales = daily_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
        
        # 5. Sort by date just in case
        daily_sales = daily_sales.sort_values('ds')
        
        # Optional: Limit data size if 'days' is provided
        if days:
            daily_sales = daily_sales.tail(days)
            
        return daily_sales

    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to simulated data if internet fails (Robustness)
        return _generate_fallback_data()

def _generate_fallback_data():
    """
    Private helper function to generate fake data if the URL fails.
    """
    import numpy as np
    dates = pd.date_range(start="2024-01-01", periods=100)
    values = np.linspace(100, 200, 100) + np.random.normal(0, 10, 100)
    return pd.DataFrame({'ds': dates, 'y': values})