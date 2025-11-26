import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

class SimpleForecaster:
    def __init__(self):
        # We initialize the model here
        self.model = LinearRegression()
        
    def train(self, df):
        """
        Trains the model on the provided dataframe.
        Expects 'ds' (date) and 'y' (value) columns.
        """
        # Convert date to ordinal (number) so regression can handle it
        df['date_ordinal'] = pd.to_datetime(df['ds']).apply(lambda x: x.toordinal())
        X = df[['date_ordinal']]
        y = df['y']
        
        self.model.fit(X, y)
        return self
        
    def predict_next_days(self, df, days_to_predict=30):
        """
        Predicts the next N days based on the training.
        """
        last_date = pd.to_datetime(df['ds'].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
        
        future_ordinals = future_dates.map(lambda x: x.toordinal()).values.reshape(-1, 1)
        predictions = self.model.predict(future_ordinals)
        
        forecast_df = pd.DataFrame({'ds': future_dates, 'y_pred': predictions})
        return forecast_df