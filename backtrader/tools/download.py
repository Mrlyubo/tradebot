import requests
import pandas as pd

# Your Tiingo API key
api_key = 'e5562fd4c66b35766597200aa8152f8ae0a15e8f'

# Define the tickers and date range
tickers = ['AMZN', 'MSFT']
start_date = '2017-01-01'
end_date = '2025-12-31'

# Dictionary to store the data for each ticker
data_dict = {}

# Loop over each ticker to download its data
for ticker in tickers:
    url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices'
    params = {
        'startDate': start_date,
        'endDate': end_date,
        'token': api_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        # Convert the JSON data to a pandas DataFrame
        data = response.json()
        df = pd.DataFrame(data)
        data_dict[ticker] = df
        print(f"Data for {ticker} downloaded successfully.")
    else:
        print(f"Error retrieving data for {ticker}: {response.status_code}")

# Example: print the first few rows for each ticker
for ticker, df in data_dict.items():
    print(f"\nTicker: {ticker}")
    print(df.head())
