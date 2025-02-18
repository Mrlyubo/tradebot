import os
import time
import requests
import pandas as pd
import yfinance as yf

# Directory for caching financials
CACHE_DIR = "../cache_financials"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_current_sp500_constituents():
    """
    Scrape the current S&P 500 constituents from Wikipedia.
    Returns a DataFrame with ['Symbol', 'Security'] columns.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url)
    df_list = pd.read_html(resp.text)  # might raise a FutureWarning
    sp500_df = df_list[0]
    # Rename columns if needed
    sp500_df.columns = [c.replace(" ", "_") for c in sp500_df.columns]
    return sp500_df[['Symbol', 'Security']].copy()

def fetch_or_load_financials(ticker, pause=1.0):
    """
    Load the annual financials for `ticker` from a local pickle if available.
    Otherwise, download from yfinance, save locally, and return the DataFrame.
    """
    local_path = os.path.join(CACHE_DIR, f"{ticker}_financials.pkl")
    
    # If file exists, load & return
    if os.path.exists(local_path):
        df_cached = pd.read_pickle(local_path)
        return df_cached
    
    # Otherwise download from yfinance
    tk = yf.Ticker(ticker)
    try:
        df_fin = tk.financials  # annual data
        time.sleep(pause)       # small delay to avoid rate limit
    except Exception as e:
        print(f"Error downloading financials for {ticker}: {e}")
        return pd.DataFrame()  # empty if fail
    
    if not df_fin.empty:
        df_fin.to_pickle(local_path)
    return df_fin

def get_3yr_revenue_growth_with_labels(ticker):
    """
    1) Loads the cached (or downloads) annual financials for ticker
    2) Finds the 'Total Revenue' (or 'Revenue') row
    3) Sorts columns ascending by date (they are Timestamps)
    4) Extracts the last 4 columns => final 3 yoy intervals
    5) Labels each yoy by the ending year (e.g. 'YoY2020'), plus an 'AvgGrowth'

    Returns a dict like:
    {
      "YoY2020": 0.25,  # 25% yoy from 2019->2020
      "YoY2021": 0.30,
      "YoY2022": 0.22,
      "AvgGrowth": 0.256...
    }
    or None if not enough data or row missing.
    """
    df_fin = fetch_or_load_financials(ticker)
    if df_fin.empty:
        return None
    
    # Attempt to find a revenue row
    possible_labels = ["Total Revenue", "Revenue"]
    revenue_series = None
    for lbl in possible_labels:
        if lbl in df_fin.index:
            revenue_series = df_fin.loc[lbl]
            break
    if revenue_series is None:
        return None

    # Sort columns ascending
    revenue_series = revenue_series.sort_index()

    # Need at least 4 data points => 3 yoy intervals
    if len(revenue_series) < 4:
        return None

    # Take last 4 => final 3 yoy intervals
    revenue_series = revenue_series.iloc[-4:]
    date_list = list(revenue_series.index)  # these are Timestamps

    yoy_values = []
    result = {}

    for i in range(len(date_list) - 1):
        start_date = date_list[i]
        end_date   = date_list[i+1]
        
        start_rev = revenue_series.loc[start_date]
        end_rev   = revenue_series.loc[end_date]
        if start_rev <= 0:
            return None
        
        yoy = (end_rev / start_rev) - 1
        end_year = end_date.year
        label = f"YoY{end_year}"
        result[label] = yoy
        yoy_values.append(yoy)
    
    if len(yoy_values) != 3:
        return None

    avg_growth = sum(yoy_values) / 3.0
    result["AvgGrowth"] = avg_growth
    return result

def main():
    # 1) Get current S&P 500 membership
    sp500_df = get_current_sp500_constituents()
    print(f"Found {len(sp500_df)} current S&P 500 tickers.")

    # 2) For each ticker, compute the last 3 yoy intervals of revenue
    results = []
    for idx, row in sp500_df.iterrows():
        ticker = row['Symbol']
        company = row['Security']

        # Update tickers like "BRK.B" -> "BRK-B" for yfinance
        if "." in ticker:
            ticker = ticker.replace(".", "-")
        
        yoy_dict = get_3yr_revenue_growth_with_labels(ticker)
        if yoy_dict is None:
            continue
        
        # Flatten yoy_dict => a single row
        data_row = {"Ticker": ticker, "Company": company}
        data_row.update(yoy_dict)
        results.append(data_row)

        # Optional short sleep if worried about rate limits
        # time.sleep(0.5)
    
    if not results:
        print("No valid data for S&P 500 tickers.")
        return
    
    # 3) Sort by AvgGrowth descending
    df_res = pd.DataFrame(results)
    df_res.sort_values("AvgGrowth", ascending=False, inplace=True)
    
    # 4) Print
    print("\n=== S&P 500 by 3-year yoy revenue growth (AvgGrowth) ===")
    for idx, row in df_res.iterrows():
        ticker = row["Ticker"]
        company = row["Company"]
        yoy_cols = [c for c in df_res.columns if c.startswith("YoY") and c != "AvgGrowth"]
        yoy_cols = sorted(yoy_cols)  # e.g. ["YoY2020", "YoY2021", "YoY2022"]
        
        yoy_strs = []
        for ccol in yoy_cols:
            pct = row[ccol] * 100
            yoy_strs.append(f"{ccol}={pct:6.1f}%")
        avg_pct = row["AvgGrowth"] * 100

        yoy_str = "  ".join(yoy_strs)
        print(f"{ticker:<8} {company[:25]:<25} {yoy_str}  Avg={avg_pct:6.1f}%")
    
    # 5) Save results
    df_res.to_csv("sp500_revenue_growth.csv", index=False)
    print("\nSaved 'sp500_revenue_growth.csv'. Cached financials in 'cache_financials/'.")
    
def get_ticker_revenue_growth(ticker, pause=1.0):
    """
    Reads the cached (or downloads) annual financials for `ticker`,
    finds the last 3 consecutive YoY revenue intervals, labeling each with the ending year,
    plus an 'AvgGrowth'.

    Returns a dict, for example:
      {
        "YoY2020": 0.25,   # => 25% yoy from 2019->2020
        "YoY2021": 0.30,
        "YoY2022": 0.22,
        "AvgGrowth": 0.2566...
      }
    or None if data is missing or not enough columns to compute.
    """
    # 1) Fetch or load the financials from local cache
    df_fin = fetch_or_load_financials(ticker, pause=pause)
    if df_fin.empty:
        return None
    
    # 2) Locate a row for revenue
    possible_labels = ["Total Revenue", "Revenue"]
    revenue_series = None
    for lbl in possible_labels:
        if lbl in df_fin.index:
            revenue_series = df_fin.loc[lbl]
            break
    if revenue_series is None:
        return None

    # 3) Sort columns ascending by date
    revenue_series = revenue_series.sort_index()

    # 4) Need at least 4 columns => 3 intervals
    if len(revenue_series) < 4:
        return None

    # Take the last 4 => final 3 yoy intervals
    revenue_series = revenue_series.iloc[-4:]

    date_list = list(revenue_series.index)  # Timestamps
    yoy_values = []
    result = {}

    for i in range(len(date_list) - 1):
        start_date = date_list[i]
        end_date   = date_list[i+1]
        start_val  = revenue_series.loc[start_date]
        end_val    = revenue_series.loc[end_date]
        
        # skip or fail if start_val <= 0
        if start_val <= 0:
            return None
        
        yoy = (end_val / start_val) - 1
        end_year = end_date.year
        label = f"YoY{end_year}"  # e.g. "YoY2022"
        result[label] = yoy
        yoy_values.append(yoy)
    
    if len(yoy_values) != 3:
        return None

    # Average yoy
    avg_growth = sum(yoy_values) / 3.0
    result["AvgGrowth"] = avg_growth
    print(result)
    return result

if __name__ == "__main__":
    main()
    # get_ticker_revenue_growth('XOM')