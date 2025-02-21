import os
import requests
import pandas as pd
import numpy as np

CACHE_DIR = "./Tiingo"
TIINGO_API_KEY = 'e5562fd4c66b35766597200aa8152f8ae0a15e8f'

# Dynamic basket mapping: each year gets its own 7-ticker basket.
top7_by_growth = {
    2017: ['AMD', 'NVDA', 'FCX', 'TRGP', 'TPL', 'OKE', 'STLD'], 
    2018: ['ALGN', 'ANET', 'TTWO', 'BA', 'NVDA', 'NVR', 'FSLR'],
    2019: ['ENPH', 'DXCM', 'AXON', 'LULU', 'KDP', 'AMD', 'FTNT'], 
    2020: ['ENPH', 'PODD', 'AMD', 'PAYC', 'LRCX', 'TER', 'BLDR'], 
    2021: ['TSLA', 'ENPH', 'MRNA', 'CRWD', 'GNRC', 'FCX', 'ALB'], 
    2022: ['DVN', 'F', 'FANG', 'FTNT', 'NVDA', 'NUE', 'BLDR'], 
    2023: ['FSLR', 'TPL', 'OXY', 'STLD', 'SMCI', 'ENPH', 'HES'], 
    2024: ['SMCI', 'NVDA', 'CRWD', 'META', 'PLTR', 'PANW', 'BLDR']
}

def download_tiingo_data(tickers, start_date="2017-01-01", end_date="2025-12-31"):
    """
    Downloads historical daily prices from Tiingo for the given tickers between start_date and end_date.
    Saves the data as CSV files in CACHE_DIR (one per ticker, with start_date and end_date in the filename)
    and returns a combined DataFrame with an added 'Ticker' column.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    all_dataframes = []
    for ticker in tickers:
        csv_filename = f"{ticker}_{start_date}_{end_date}.csv"
        csv_path = os.path.join(CACHE_DIR, csv_filename)
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
    Converts 'date' to datetime, sets it as index, sorts the DataFrame,
    filters to only include rows between start_date and end_date,
    and selects the price column.
    """
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[start_date:end_date]
    price_col = 'adjClose' if 'adjClose' in df.columns else 'close'
    return df, price_col

def compute_monthly_metrics_for_ticker(df, ticker, price_col, investment_per_stock, start_date, end_date):
    """
    For a single ticker (used for the overall mag7 ETF simulation), resample daily data to monthly
    (using the first trading day) and compute shares bought, invested amount, cumulative invested,
    cumulative shares, and portfolio value. The monthly index is forced to span exactly from start_date to end_date.
    """
    df_ticker = df[df["Ticker"] == ticker]
    monthly = df_ticker.resample('MS').first()
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS', tz=monthly.index.tz)
    monthly = monthly.reindex(monthly_index, method='ffill')
    
    monthly['shares_bought'] = investment_per_stock / monthly[price_col]
    monthly['invested'] = investment_per_stock
    monthly['cumulative_invested'] = monthly['invested'].cumsum()
    monthly['cumulative_shares'] = monthly['shares_bought'].cumsum()
    monthly['portfolio_value'] = monthly['cumulative_shares'] * monthly[price_col]
    return monthly

def compute_monthly_metrics_for_basket(df, ticker, price_col, investment_per_stock, basket_start, basket_end, global_end):
    """
    For a given ticker in a basket (dynamic basket simulation), resample daily data (filtered to [basket_start, global_end])
    to monthly frequency (using the first trading day). For months in the active period [basket_start, basket_end],
    the basket invests a fixed amount each month; after basket_end, no additional cash is invested (invested=0).
    Shares are only purchased during the active period. The portfolio value is updated over time.
    """
    df_ticker = df[df["Ticker"] == ticker]
    monthly = df_ticker.resample('MS').first()
    # Use the timezone of monthly data (if any)
    tz = monthly.index.tz
    overall_index = pd.date_range(start=basket_start, end=global_end, freq='MS', tz=tz)
    monthly = monthly.reindex(overall_index, method='ffill')
    
    # Set invested and new shares bought only during the active period.
    monthly['invested'] = investment_per_stock
    basket_end_dt = pd.Timestamp(basket_end)
    if tz is not None:
        basket_end_dt = basket_end_dt.tz_localize(tz)
    monthly.loc[monthly.index > basket_end_dt, 'invested'] = 0
    monthly['shares_bought'] = investment_per_stock / monthly[price_col]
    monthly.loc[monthly.index > basket_end_dt, 'shares_bought'] = 0
    
    monthly['cumulative_invested'] = monthly['invested'].cumsum()
    monthly['cumulative_shares'] = monthly['shares_bought'].cumsum()
    monthly['portfolio_value'] = monthly['cumulative_shares'] * monthly[price_col]
    return monthly

def max_drawdown(series):
    """Calculates maximum drawdown from a portfolio value series."""
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return drawdown.min()

def calculate_avg_annual_return(invested, final_value, years):
    """
    Computes the average annual return (CAGR) given total invested capital,
    the final portfolio value, and the number of years.
    """
    if invested <= 0 or years <= 0:
        return 0.0
    return ((final_value / invested) ** (1 / years) - 1) * 100

def compute_yearly_metrics_for_ticker(monthly):
    """
    Resamples monthly data to yearly frequency (last month of each year) and computes:
      - percent gain,
      - cumulative gain,
      - maximum drawdown,
      - average annual return (CAGR) from the first investment.
    """
    yearly = monthly.resample('YE').last()
    yearly['profit_rate'] = (yearly['portfolio_value'] - yearly['cumulative_invested']) / yearly['cumulative_invested']
    yearly['cumulative_gain'] = yearly['portfolio_value'] - yearly['cumulative_invested']
    
    max_dd = monthly.groupby(monthly.index.year)['portfolio_value'].apply(max_drawdown)
    yearly['max_drawdown'] = max_dd.values
    
    yearly['percent_gain'] = (yearly['profit_rate'] * 100).map(lambda x: f"{x:.2f}%")
    yearly['max_drawdown'] = (yearly['max_drawdown'] * 100).map(lambda x: f"{x:.2f}%")
    
    first_date = monthly.index.min()
    avg_returns = []
    for date, row in yearly.iterrows():
        years_elapsed = (date - first_date).days / 365.25
        avg_return = calculate_avg_annual_return(row['cumulative_invested'], row['portfolio_value'], years_elapsed)
        avg_returns.append(f"{avg_return:.2f}%")
    yearly['avg_annual_return'] = avg_returns
    return yearly

def aggregate_portfolio(monthly_dict, start_date, end_date):
    """
    Aggregates monthly metrics for all tickers (used for overall mag7 ETF simulation).
    For each month, sums the invested amounts and portfolio values across tickers.
    The index is forced to be a complete monthly date range from start_date to end_date.
    """
    sample = list(monthly_dict.values())[0]
    tz = sample.index.tz
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS', tz=tz)
    portfolio = pd.DataFrame(index=monthly_index)
    portfolio["invested"] = 0.0
    portfolio["portfolio_value"] = 0.0

    for ticker, monthly in monthly_dict.items():
        m_inv = monthly["invested"].reindex(monthly_index, method='ffill').fillna(0)
        m_value = monthly["portfolio_value"].reindex(monthly_index, method='ffill').fillna(0)
        portfolio["invested"] += m_inv
        portfolio["portfolio_value"] += m_value
        
    portfolio["cumulative_invested"] = portfolio["invested"].cumsum()
    return portfolio.sort_index()

def aggregate_portfolios(portfolio_list, global_start, global_end):
    """
    Aggregates multiple monthly portfolio series (from different baskets) by summing their
    'invested' and 'portfolio_value' columns. The output is a combined portfolio series
    spanning the global period.
    """
    tz = portfolio_list[0].index.tz
    monthly_index = pd.date_range(start=global_start, end=global_end, freq='MS', tz=tz)
    overall = pd.DataFrame(index=monthly_index)
    overall["invested"] = 0.0
    overall["portfolio_value"] = 0.0
    
    for port in portfolio_list:
        p_inv = port["invested"].reindex(monthly_index, method='ffill').fillna(0)
        p_val = port["portfolio_value"].reindex(monthly_index, method='ffill').fillna(0)
        overall["invested"] += p_inv
        overall["portfolio_value"] += p_val
    
    overall["cumulative_invested"] = overall["invested"].cumsum()
    return overall.sort_index()

def compute_yearly_metrics_for_portfolio(portfolio):
    """Computes yearly metrics for an aggregated portfolio series."""
    return compute_yearly_metrics_for_ticker(portfolio)

def main():
    # Overall simulation period for the static mag7 ETF:
    static_start = "2017-01-01"
    static_end = "2024-12-31"
    total_investment = 1500   # total monthly investment
    investment_per_stock = total_investment / 7
    mag7_tickers = ["AAPL", "AMZN", "NVDA", "GOOG", "META", "TSLA", "MSFT"]
    
    # Process static mag7 ETF (buy same stocks every month)
    df_static = download_tiingo_data(mag7_tickers, start_date=static_start, end_date=static_end)
    df_static, price_col_static = prepare_data(df_static, static_start, static_end)
    
    monthly_dict = {}
    print("Individual Ticker Metrics (Static mag7):")
    for ticker in mag7_tickers:
        monthly = compute_monthly_metrics_for_ticker(df_static, ticker, price_col_static, investment_per_stock, static_start, static_end)
        monthly_dict[ticker] = monthly
        yearly = compute_yearly_metrics_for_ticker(monthly)
        print(f"\nYearly Metrics for {ticker}:")
        print(yearly[['cumulative_invested', 'cumulative_gain', 'portfolio_value', 
                      'percent_gain', 'avg_annual_return', 'max_drawdown']])
    
    portfolio_static = aggregate_portfolio(monthly_dict, static_start, static_end)
    yearly_static = compute_yearly_metrics_for_portfolio(portfolio_static)
    
    print("\nAggregated Yearly Metrics for Static mag7 ETF:")
    print(yearly_static[['cumulative_invested', 'cumulative_gain', 'portfolio_value', 
                         'percent_gain', 'avg_annual_return', 'max_drawdown']])
    
    # Now, process the dynamic baskets (buy new basket each year and hold forever).
    global_start = "2017-01-01"
    global_end = "2024-12-31"
    total_monthly_investment = 1500
    overall_portfolio_series = []
    
    for year in sorted(top7_by_growth.keys()):
        basket_start = f"{year}-01-01"
        basket_end = f"{year}-12-31"  # active investment period for the basket
        tickers = top7_by_growth[year]
        investment_per_stock = total_monthly_investment / 7
        
        # print(f"\n==== Processing Basket for Year: {year} (Hold Forever) ====")
        df_basket = download_tiingo_data(tickers, start_date=basket_start, end_date=global_end)
        df_basket, price_col_basket = prepare_data(df_basket, basket_start, global_end)
        
        basket_monthly_series = []
        for ticker in tickers:
            monthly = compute_monthly_metrics_for_basket(df_basket, ticker, price_col_basket, investment_per_stock, basket_start, basket_end, global_end)
            basket_monthly_series.append(monthly)
        
        # Manually aggregate the basket: sum invested and portfolio_value across its tickers.
        monthly_index = basket_monthly_series[0].index
        basket_agg = pd.DataFrame(index=monthly_index)
        basket_agg["invested"] = sum(m["invested"].reindex(monthly_index, method='ffill').fillna(0) for m in basket_monthly_series)
        basket_agg["portfolio_value"] = sum(m["portfolio_value"].reindex(monthly_index, method='ffill').fillna(0) for m in basket_monthly_series)
        basket_agg["cumulative_invested"] = basket_agg["invested"].cumsum()
        
        overall_portfolio_series.append(basket_agg)
    
    overall_portfolio = aggregate_portfolios(overall_portfolio_series, global_start, global_end)
    yearly_overall = compute_yearly_metrics_for_portfolio(overall_portfolio)
    
    print("\n==== Aggregated Yearly Metrics for the Top7Grow ETF Portfolio ====")
    print(yearly_overall[['cumulative_invested', 'cumulative_gain', 'portfolio_value', 
                            'percent_gain', 'avg_annual_return', 'max_drawdown']])
    
if __name__ == "__main__":
    main()
