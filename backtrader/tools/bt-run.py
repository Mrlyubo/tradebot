import os
import requests
import pandas as pd
import numpy as np

CACHE_DIR = "./Tiingo"
TIINGO_API_KEY = 'e5562fd4c66b35766597200aa8152f8ae0a15e8f'

def download_tiingo_data(tickers, start_date="2017-01-01", end_date="2025-12-31"):
    """
    Downloads historical daily prices from Tiingo for the given tickers between start_date and end_date.
    Saves the data as CSV files in the CACHE_DIR (one per ticker) and returns a combined DataFrame
    with an additional column 'Ticker'.
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

def prepare_data(df, start_date, end_date):
    """
    Sets the date as the index, sorts the DataFrame, filters it to only include data
    between start_date and end_date, and chooses the appropriate price column.
    """
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    # Filter the DataFrame between start_date and end_date
    df = df.loc[start_date:end_date]
    # Use adjusted close if available (to account for splits)
    price_col = 'adjClose' if 'adjClose' in df.columns else 'close'
    return df, price_col

def compute_monthly_metrics_for_ticker(df, ticker, price_col, investment_per_stock):
    """
    Filters the DataFrame for a single ticker, resamples to monthly (using the first trading day),
    and computes:
      - shares bought in the month,
      - cumulative invested capital,
      - cumulative shares,
      - portfolio value.
    """
    df_ticker = df[df["Ticker"] == ticker]
    monthly = df_ticker.resample('MS').first()
    monthly['shares_bought'] = investment_per_stock / monthly[price_col]
    monthly['invested'] = investment_per_stock
    monthly['cumulative_invested'] = monthly['invested'].cumsum()
    monthly['cumulative_shares'] = monthly['shares_bought'].cumsum()
    monthly['portfolio_value'] = monthly['cumulative_shares'] * monthly[price_col]
    return monthly

def max_drawdown(series):
    """
    Calculates the maximum drawdown from a portfolio value series.
    """
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return drawdown.min()

def calculate_avg_annual_return(invested, final_value, years):
    """
    Computes the average annual return (CAGR) given the total invested capital,
    the final portfolio value, and the number of years.
    """
    if invested <= 0 or years <= 0:
        return 0.0
    return ((final_value / invested) ** (1 / years) - 1) * 100

def compute_yearly_metrics_for_ticker(monthly):
    """
    Resamples the monthly data to yearly frequency (last month of each year)
    and computes:
      - percent gain (formatted as a percentage),
      - cumulative gain (dollar gain),
      - maximum drawdown (formatted as a percentage),
      - average annual return (CAGR) from the first investment.
    """
    yearly = monthly.resample('YE').last()
    yearly['profit_rate'] = (yearly['portfolio_value'] - yearly['cumulative_invested']) / yearly['cumulative_invested']
    yearly['cumulative_gain'] = yearly['portfolio_value'] - yearly['cumulative_invested']
    
    max_dd = monthly.groupby(monthly.index.year)['portfolio_value'].apply(max_drawdown)
    yearly['max_drawdown'] = max_dd.values
    
    yearly['percent_gain'] = (yearly['profit_rate'] * 100).map(lambda x: f"{x:.2f}%")
    yearly['max_drawdown'] = (yearly['max_drawdown'] * 100).map(lambda x: f"{x:.2f}%")
    
    # Calculate average annual return (CAGR) from the first investment date to each year-end.
    first_date = monthly.index.min()
    avg_returns = []
    for date, row in yearly.iterrows():
        years_elapsed = (date - first_date).days / 365.25
        avg_return = calculate_avg_annual_return(row['cumulative_invested'], row['portfolio_value'], years_elapsed)
        avg_returns.append(f"{avg_return:.2f}%")
    yearly['avg_annual_return'] = avg_returns
    return yearly

def aggregate_portfolio(monthly_dict):
    """
    Aggregates monthly metrics for all tickers. For each month, sums the invested amounts
    and portfolio values across tickers. Recalculates cumulative invested as the cumulative
    sum of the monthly invested.
    """
    # Initialize index from the first ticker's monthly data
    first_index = list(monthly_dict.values())[0].index
    all_index = first_index.copy()
    for m in monthly_dict.values():
        all_index = all_index.union(m.index)
    
    portfolio = pd.DataFrame(index=all_index)
    portfolio["invested"] = 0.0
    portfolio["portfolio_value"] = 0.0

    for ticker, monthly in monthly_dict.items():
        m_inv = monthly["invested"].reindex(all_index, method='ffill').fillna(0)
        m_value = monthly["portfolio_value"].reindex(all_index, method='ffill').fillna(0)
        portfolio["invested"] += m_inv
        portfolio["portfolio_value"] += m_value
        
    portfolio["cumulative_invested"] = portfolio["invested"].cumsum()
    return portfolio.sort_index()

def compute_yearly_metrics_for_portfolio(portfolio):
    """
    Resamples the overall monthly portfolio data to yearly frequency and computes:
      - percent gain,
      - cumulative gain,
      - maximum drawdown,
      - average annual return.
    """
    yearly = portfolio.resample('YE').last()
    yearly['profit_rate'] = (yearly['portfolio_value'] - yearly['cumulative_invested']) / yearly['cumulative_invested']
    yearly['cumulative_gain'] = yearly['portfolio_value'] - yearly['cumulative_invested']
    
    max_dd = portfolio.groupby(portfolio.index.year)['portfolio_value'].apply(max_drawdown)
    yearly['max_drawdown'] = max_dd.values
    
    yearly['percent_gain'] = (yearly['profit_rate'] * 100).map(lambda x: f"{x:.2f}%")
    yearly['max_drawdown'] = (yearly['max_drawdown'] * 100).map(lambda x: f"{x:.2f}%")
    
    first_date = portfolio.index.min()
    avg_returns = []
    for date, row in yearly.iterrows():
        years_elapsed = (date - first_date).days / 365.25
        avg_return = calculate_avg_annual_return(row['cumulative_invested'], row['portfolio_value'], years_elapsed)
        avg_returns.append(f"{avg_return:.2f}%")
    yearly['avg_annual_return'] = avg_returns
    return yearly

def main():
    # Define the "mag7" ETF tickers and parameters
    tickers = ["AAPL", "AMZN", "NVDA", "GOOG", "META", "TSLA", "MSFT"]
    start_date = "2017-01-01"
    end_date = "2024-12-31"
    total_investment = 1500   # total monthly investment for the ETF
    investment_per_stock = total_investment / len(tickers)
    
    # Download and prepare the data for all tickers
    df = download_tiingo_data(tickers, start_date=start_date, end_date=end_date)
    df, price_col = prepare_data(df, start_date, end_date)
    
    monthly_dict = {}
    print("Individual Ticker Metrics:")
    for ticker in tickers:
        monthly = compute_monthly_metrics_for_ticker(df, ticker, price_col, investment_per_stock)
        monthly_dict[ticker] = monthly
        yearly = compute_yearly_metrics_for_ticker(monthly)
        print(f"\nYearly Metrics for {ticker}:")
        print(yearly[['cumulative_invested', 'cumulative_gain', 'portfolio_value', 
                      'percent_gain', 'avg_annual_return', 'max_drawdown']])
    
    # Aggregate metrics for the overall mag7 ETF
    portfolio = aggregate_portfolio(monthly_dict)
    yearly_portfolio = compute_yearly_metrics_for_portfolio(portfolio)
    
    print("\nAggregated Yearly Metrics for mag7 ETF:")
    print(yearly_portfolio[['cumulative_invested', 'cumulative_gain', 'portfolio_value', 
                            'percent_gain', 'avg_annual_return', 'max_drawdown']])
    
if __name__ == "__main__":
    main()
