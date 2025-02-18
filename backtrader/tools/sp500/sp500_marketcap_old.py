# import requests
# import pandas as pd
# import yfinance as yf
# import datetime
# import time
# import os

# DATA_DIR = "../data_cache"
# os.makedirs(DATA_DIR, exist_ok=True)

# def fetch_or_load_yf_data(ticker, start_date, end_date, pause=1.0):
#     """
#     Check if we already have a local CSV for (ticker, start_date, end_date).
#     If so, load and return it. Otherwise, download from Yahoo, save, then return.
#     """
#     file_name = f"{ticker}_{start_date}_{end_date}.csv".replace(":", "-")
#     local_path = os.path.join(DATA_DIR, file_name)
    
#     # if os.path.exists(local_path):
#     #     df = pd.read_csv(local_path, parse_dates=True, index_col=0)
#     #     return df
#     # else:
#     print(f"Downloading {ticker} from {start_date} to {end_date} ...")
#     df = yf.download(ticker, start=start_date, end=end_date, progress=False)
#     time.sleep(pause)
#     df.to_csv(local_path)
#     return df

# def get_current_sp500_constituents():
#     """
#     Scrape current S&P 500 constituents from Wikipedia.
#     Returns a DataFrame with 'Symbol' and 'Security' (company name).
#     """
#     url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#     resp = requests.get(url)
#     sp500_list = pd.read_html(resp.text)[0]
#     sp500_list.columns = [col.replace(' ', '_') for col in sp500_list.columns]
#     return sp500_list[['Symbol', 'Security']].copy()

# def get_year_end_market_cap(ticker, years, pause=1.0):
#     """
#     Fetch approximate year-end market cap for the given ticker,
#     returning a dictionary {year: market_cap}.
    
#     - ticker: Stock ticker symbol
#     - years: List of integer years, e.g. [2015, 2016, 2017, ...]
#     - pause: Optional delay (seconds) for rate-limiting
    
#     Example return:
#       {2015: 4.6e10, 2016: 6.0e10, ...}
#     or {year: None} if data unavailable.
#     """
#     results = {}
    
#     # 1) Fetch general ticker info once
#     ticker_yf = yf.Ticker(ticker)
#     ticker_info = ticker_yf.info
#     shares_outstanding = ticker_info.get('sharesOutstanding', None)
    
#     for y in years:
#         # 2) Define date range around year-end
#         start_date = datetime.date(y, 12, 15).strftime("%Y-%m-%d")
#         end_date = datetime.date(y+1, 1, 15).strftime("%Y-%m-%d")
        
#         # 3) Download (or load from cache) the daily data in that date range
#         df_price = fetch_or_load_yf_data(ticker, start_date, end_date, pause=pause)
#         if df_price.empty:
#             results[y] = None
#             continue
        
#         # 4) Identify the last trading day
#         last_day = df_price.index.max()
#         if pd.isnull(last_day):
#             results[y] = None
#             continue
        
#         close_price = df_price.loc[last_day, 'Close']
        
#         # 5) Compute approximate market cap
#         if shares_outstanding is not None and shares_outstanding > 0:
#             mcap = close_price * shares_outstanding
#             if isinstance(mcap, pd.Series) and len(mcap) == 1:
#                 mcap = mcap.iloc[0]
#             results[y] = mcap if pd.notnull(mcap) else None
#         else:
#             results[y] = None
    
#     return results

# def main():
#     # 1) Get current S&P 500 constituents
#     sp500_df = get_current_sp500_constituents()
#     # For testing, let's just take a small sample
#     # sp500_df = sp500_df.head()
#     print(f"Found {len(sp500_df)} tickers (current S&P 500).")
    
#     # 2) Define years we want
#     years = list(range(2015, 2026))  # e.g. 2015 -> 2025
    
#     # 3) For each ticker, fetch year-end market caps
#     all_data = []
    
#     for i, row in sp500_df.iterrows():
#         ticker = row['Symbol']
#         company = row['Security']
        
#         # Handle tickers like BRK.B -> BRK-B
#         if "." in ticker:
#             ticker = ticker.replace(".", "-")
        
#         mcaps_dict = get_year_end_market_cap(ticker, years, pause=0.5)
        
#         # Flatten out results into a list of rows
#         for y in years:
#             all_data.append({
#                 'Ticker': ticker,
#                 'Company': company,
#                 'Year': y,
#                 'MarketCap': mcaps_dict.get(y, None)
#             })
        
#         print(f"Processed: {ticker}")
    
#     # 4) Convert results to DataFrame and save
#     results_df = pd.DataFrame(all_data)
#     results_df.to_csv("sp500_year_end_marketcaps.csv", index=False)
#     print("Saved sp500_year_end_marketcaps.csv")

# if __name__ == "__main__":
#     main()
