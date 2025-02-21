import os
import requests
import pandas as pd
import numpy as np

CACHE_DIR = "./Tiingo"
TIINGO_API_KEY = 'e5562fd4c66b35766597200aa8152f8ae0a15e8f'

def download_tiingo_data(tickers, start_date="2017-01-01", end_date="2025-12-31"):
    """
    Downloads historical daily prices from Tiingo for the given ticker(s)
    between start_date and end_date (YYYY-MM-DD format).
    Saves the data as CSV in the CACHE_DIR folder (one file per ticker).
    Returns a combined DataFrame with a 'Ticker' column.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    all_dataframes = []
    for ticker in tickers:
        csv_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        if os.path.exists(csv_path):
            df_ticker = pd.read_csv(csv_path, parse_dates=["date"])
        else:
            print(f"Downloading data for {ticker} from Tiingo...")
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "token": TIINGO_API_KEY,
                "format": "json"
            }
            response = requests.get(url, params=params)
            data = response.json()
            if not isinstance(data, list):
                raise ValueError(f"Error fetching data for {ticker}: {data}")
            df_ticker = pd.DataFrame(data)
            if df_ticker.empty:
                raise ValueError(f"No data returned for {ticker} in range {start_date} to {end_date}.")
            df_ticker["date"] = pd.to_datetime(df_ticker["date"]).dt.normalize()
            df_ticker = df_ticker.sort_values("date").reset_index(drop=True)
            df_ticker.to_csv(csv_path, index=False)
        
        df_ticker["Ticker"] = ticker
        all_dataframes.append(df_ticker)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

def prepare_data(df):
    """
    Prepares the DataFrame by setting the date as index, sorting, and
    choosing the appropriate price column (adjClose if available).
    """
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    price_col = 'adjClose' if 'adjClose' in df.columns else 'close'
    return df, price_col

def compute_monthly_metrics(df, price_col, investment=1500):
    """
    Resamples the daily data to monthly data (first trading day of each month),
    calculates shares bought, cumulative shares, invested capital, and portfolio value.
    """
    monthly = df.resample('MS').first()
    monthly['shares_bought'] = investment / monthly[price_col]
    monthly['invested'] = investment
    monthly['cumulative_shares'] = monthly['shares_bought'].cumsum()
    monthly['cumulative_invested'] = monthly['invested'].cumsum()
    monthly['portfolio_value'] = monthly['cumulative_shares'] * monthly[price_col]
    return monthly

def max_drawdown(series):
    """
    Calculates maximum drawdown for a given portfolio value series.
    """
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return drawdown.min()

def compute_yearly_metrics(monthly):
    """
    Resamples the monthly data to yearly frequency (last month of each year) and computes:
    - percent gain (as formatted percentage)
    - cumulative gain (raw dollar difference)
    - max drawdown (formatted percentage)
    """
    yearly = monthly.resample('YE').last()
    # Calculate profit rate and cumulative gain
    yearly['profit_rate'] = (yearly['portfolio_value'] - yearly['cumulative_invested']) / yearly['cumulative_invested']
    yearly['cumulative_gain'] = yearly['portfolio_value'] - yearly['cumulative_invested']
    
    # Compute max drawdown by grouping monthly data by year
    max_dd = monthly.groupby(monthly.index.year)['portfolio_value'].apply(max_drawdown)
    yearly['max_drawdown'] = max_dd.values  # align by year

    # Format percent gain and max drawdown
    yearly['percent_gain'] = (yearly['profit_rate'] * 100).map(lambda x: f"{x:.2f}%")
    yearly['max_drawdown'] = (yearly['max_drawdown'] * 100).map(lambda x: f"{x:.2f}%")
    return yearly

def main():
    # Parameters
    ticker = "AMZN"
    start_date = "2017-01-01"
    end_date = "2025-01-01"
    investment = 1500

    # Download and prepare data
    df = download_tiingo_data(ticker, start_date=start_date, end_date=end_date)
    df, price_col = prepare_data(df)
    
    # Compute monthly and yearly metrics
    monthly = compute_monthly_metrics(df, price_col, investment)
    yearly = compute_yearly_metrics(monthly)
    
    # Print yearly metrics
    print("Yearly Metrics:")
    print(yearly[['cumulative_invested', 'cumulative_gain', 'portfolio_value', 'percent_gain', 'max_drawdown']])

if __name__ == "__main__":
    main()
