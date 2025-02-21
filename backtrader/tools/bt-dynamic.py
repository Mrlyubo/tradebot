import os
import requests
import pandas as pd
from datetime import date, datetime, timedelta
# Replace this with your actual Tiingo API key
TIINGO_API_KEY = "e5562fd4c66b35766597200aa8152f8ae0a15e8f"

# Directory for caching
CACHE_DIR = "Tiingo"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


########################################################
# 1) Download and Cache All Ticker Data to Local CSVs  #
########################################################

def download_tiingo_data(ticker, start_date="2017-01-01", end_date="2025-12-31"):
    """
    Downloads historical daily prices from Tiingo for the given ticker 
    between start_date and end_date (YYYY-MM-DD format).
    Saves the data as CSV in the CACHE_DIR folder.
    Returns a pandas DataFrame with columns: date, adjClose, etc.
    """
    # Path to cached CSV
    csv_path = os.path.join(CACHE_DIR, f"{ticker}.csv")

    # Check if CSV already exists
    if os.path.exists(csv_path):
        # Load existing data
        df = pd.read_csv(csv_path, parse_dates=["date"])
        # Optional: Check if the date range is sufficient
        # For simplicity, let's just use what we have
        return df
    
    # Otherwise, request from Tiingo
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
        # Possibly an error message if the ticker is invalid
        raise ValueError(f"Error fetching data for {ticker}: {data}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    if df.empty:
        # No data returned
        raise ValueError(f"No data returned for {ticker} in range {start_date} to {end_date}.")

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Sort by date just in case
    df = df.sort_values("date").reset_index(drop=True)

    # Save to CSV
    df.to_csv(csv_path, index=False)
    return df


#############################################
# 2) Helpers to Get Prices from the Cached  #
#############################################

def get_local_data(ticker):
    """
    Loads the cached CSV for the given ticker and returns a DataFrame
    with a DateTimeIndex, sorted by date, for quick lookups.
    """
    csv_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        # If not downloaded yet, do it now.
        df = download_tiingo_data(ticker)
    else:
        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    df.set_index("date", inplace=True)
    return df

def get_price_on_or_after(ticker_df, target_date):
    """
    Given a ticker's DataFrame (with a DateTimeIndex) and a target_date (a date object),
    returns the 'adjClose' price on that date if available, or on the next available trading day.
    """
    # Convert target_date to a UTC-aware Timestamp
    ts = pd.Timestamp(target_date, tz='UTC')
    # Now the comparison is between tz-aware objects
    valid_dates = ticker_df.index[ticker_df.index >= ts]
    if len(valid_dates) == 0:
        return None  # no data on or after target_date
    chosen_date = valid_dates[0]
    row = ticker_df.loc[chosen_date]
    return row["adjClose"]

def get_price_on_or_before(ticker_df, target_date):
    """
    Returns the 'adjClose' price on or before target_date.
    """
    ts = pd.Timestamp(target_date, tz='UTC')
    valid_dates = ticker_df.index[ticker_df.index <= ts]
    if len(valid_dates) == 0:
        return None  # no data on or before target_date
    chosen_date = valid_dates[-1]
    row = ticker_df.loc[chosen_date]
    return row["adjClose"]



###############################################
# 3) Main Simulation with Annual Rebalancing  #
###############################################

top7_by_growth = {
    2017: ['AMD', 'NVDA', 'FCX', 'TRGP', 'TPL', 'OKE', 'STLD'],
    # 2018: ['ALGN', 'ANET', 'TTWO', 'BA', 'NVDA', 'NVR', 'FSLR'],
    # 2019: ['ENPH', 'DXCM', 'AXON', 'LULU', 'KDP', 'AMD', 'FTNT'],
    # 2020: ['ENPH', 'PODD', 'AMD', 'PAYC', 'LRCX', 'TER', 'BLDR'],
    # 2021: ['TSLA', 'ENPH', 'MRNA', 'CRWD', 'GNRC', 'FCX', 'ALB'],
    # 2022: ['DVN', 'F', 'FANG', 'FTNT', 'NVDA', 'NUE', 'BLDR'],
    # 2023: ['FSLR', 'TPL', 'OXY', 'STLD', 'SMCI', 'ENPH', 'HES'],
    # 2024: ['SMCI', 'NVDA', 'CRWD', 'META', 'PLTR', 'PANW', 'BLDR'],
    # 2025: ['VST', 'PLTR', 'UAL', 'TPL', 'CEG', 'TRGP', 'NVDA']
}

# STEP A: Download/cache all data for all tickers first, so we don't do it repeatedly.
all_tickers = set()
for ylist in top7_by_growth.values():
    all_tickers.update(ylist)

for tkr in all_tickers:
    # Download once and cache
    download_tiingo_data(tkr, start_date="2017-01-01", end_date="2025-12-31")

# STEP B: Load each ticker's data into memory (optional, or load on demand).
ticker_dfs = {}
for tkr in all_tickers:
    ticker_dfs[tkr] = get_local_data(tkr)

############################
# 4) Run the Investment    #
############################

monthly_investment = 1000.0
all_years = list(range(2017, 2026))  # 2017..2025

total_contributed = 0.0
end_of_year_cash = 0.0

for year in all_years:
    # Annual picks
    stocks = top7_by_growth[year]
    # Initialize shares for each ticker
    shares_held = {s: 0.0 for s in stocks}

    # Buy monthly from Jan..Dec
    for month in range(1, 13):
        # If the date is beyond today's real date, you won't have data
        # (For future 2025 data, might be partial or none.)
        buy_date = date(year, month, 1)
        
        # We'll buy on or after the 1st of the month (first trading day)
        # If there's no valid trading day, skip
        for ticker in stocks:
            df = ticker_dfs[ticker]
            price = get_price_on_or_after(df, buy_date)
            if price is not None and price > 0:
                amt_per_stock = monthly_investment / len(stocks)
                shares_held[ticker] += amt_per_stock / price
        
        total_contributed += monthly_investment

    # At year end, we sell everything (annual rebalancing approach).
    sell_date = date(year, 12, 31)
    year_end_value = 0.0
    for ticker, num_shares in shares_held.items():
        df = ticker_dfs[ticker]
        sell_price = get_price_on_or_before(df, sell_date)
        if sell_price is not None and sell_price > 0:
            year_end_value += num_shares * sell_price
    
    end_of_year_cash = year_end_value

final_value = end_of_year_cash
final_profit = final_value - total_contributed

print("==== FINAL RESULTS ====")
print(f"Total Contributed:  ${total_contributed:,.2f}")
print(f"Final Portfolio Value:  ${final_value:,.2f}")
print(f"Profit:  ${final_profit:,.2f}")
