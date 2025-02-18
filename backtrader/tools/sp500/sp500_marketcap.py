import requests
import pandas as pd
import yfinance as yf
import datetime
import time
import os

DATA_DIR = "../data_cache"  # Folder to store cached CSVs locally
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_or_load_yf_data(ticker, start_date, end_date, pause=1.0):
    """
    Check if we already have a local CSV for (ticker, start_date, end_date).
    If so, load and return it.
    If not, download from Yahoo, save as CSV, then return the DataFrame.
    """
    # Make a file name like 'data_cache/AMZN_2015-01-01_2025-01-01.csv'
    file_name = f"{ticker}_{start_date}_{end_date}.csv".replace(":", "-")
    local_path = os.path.join(DATA_DIR, file_name)
    
    # 1) Check if local file exists
    if os.path.exists(local_path):
        # Load from CSV
        df = pd.read_csv(local_path, parse_dates=True, index_col=0)
        # Optional: Validate the date range if you want to ensure it's not stale
        # For example, you can check df.index.min(), df.index.max(), etc.
        return df
    else:
        # 2) Download from yfinance
        print(f"Downloading {ticker} from {start_date} to {end_date} ...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        time.sleep(pause)
        
        # 3) Save to local CSV
        df.to_csv(local_path)
        return df

def get_year_end_market_cap(ticker, year, pause=1.0):
    """
    Fetch the approximate year-end market cap for a given ticker and a single year.
    
    - ticker: stock ticker symbol (string)
    - year: an integer year, e.g. 2021
    - pause: optional time (seconds) to wait between data fetches (for rate limiting)
    
    Returns a single float (market cap in USD) or None if not available.
    """
    # 1) Fetch general ticker info
    ticker_yf = yf.Ticker(ticker)
    ticker_info = ticker_yf.info
    shares_outstanding = ticker_info.get('sharesOutstanding', None)
    
    # 2) Define date range around year-end
    start_date = datetime.date(year, 12, 15).strftime("%Y-%m-%d")
    end_date = datetime.date(year+1, 1, 15).strftime("%Y-%m-%d")
    
    # 3) Download or load from cache the daily data in that date range
    df_price = fetch_or_load_yf_data(ticker, start_date, end_date, pause=pause)
    if df_price.empty:
        return None
    
    # 4) Identify the last trading day in that range
    last_day = df_price.index.max()
    if pd.isnull(last_day):
        return None
    
    close_price = df_price.loc[last_day, 'Close']
    
    # 5) Compute market cap if we have a valid sharesOutstanding
    if shares_outstanding is not None and shares_outstanding > 0:
        mcap = close_price * shares_outstanding
        return float(mcap)
    else:
        return None

def get_current_sp500_constituents():
    """Scrape current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url)
    sp500_list = pd.read_html(resp.text)[0]  # The first table
    sp500_list.columns = [col.replace(' ', '_') for col in sp500_list.columns]
    return sp500_list[['Symbol', 'Security']].copy()

def main():
    sp500_df = get_current_sp500_constituents()
    sp10_df = sp500_df.head()
    print(f"Found {len(sp500_df)} tickers (current S&P 500).")
    
    years = list(range(2015, 2026))  # Example: 2015->2026
    all_data = []
    
    for i, row in sp500_df.iterrows():
        ticker = row['Symbol']
        company = row['Security']
        
        if "." in ticker:
            # e.g., "BRK.B" -> "BRK-B"
            ticker = ticker.replace(".", "-")
        
        mcaps = get_year_end_market_cap(ticker, years, pause=0.3)
        
        for y in years:
            all_data.append({
                'Ticker': ticker,
                'Company': company,
                'Year': y,
                'MarketCap': mcaps.get(y, None)
            })
        
        print(f"Processed: {ticker}")
    
    results_df = pd.DataFrame(all_data)
    results_df.to_csv("sp500_year_end_marketcaps.csv", index=False)
    print("Saved sp500_year_end_marketcaps.csv")

if __name__ == "__main__":
    main()
