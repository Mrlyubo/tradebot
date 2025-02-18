import os
import time
import pandas as pd
import yfinance as yf

# Directory for cached financials
CACHE_DIR = "cache_financials"
os.makedirs(CACHE_DIR, exist_ok=True)

# The Magnificent 7 tickers
MAG7_TICKERS = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "TSLA", "NVDA"]

def fetch_or_load_financials(ticker, pause=1.0):
    """
    Load the annual financials DataFrame for `ticker` from local pickle if available.
    Otherwise, download from yfinance, save as a pickle, and return the DataFrame.
    """
    # File name for pickle
    local_path = os.path.join(CACHE_DIR, f"{ticker}_financials.pkl")

    # If file exists, load & return
    if os.path.exists(local_path):
        df_cached = pd.read_pickle(local_path)
        return df_cached

    # Otherwise, download from yfinance
    tk = yf.Ticker(ticker)
    try:
        df_fin = tk.financials  # annual financial data
        time.sleep(pause)
    except Exception as e:
        print(f"Error downloading financials for {ticker}: {e}")
        return pd.DataFrame()  # empty

    # Cache if not empty
    if not df_fin.empty:
        df_fin.to_pickle(local_path)

    return df_fin

def get_3yr_revenue_growth_with_labels(ticker):
    """
    Use cached financials to find the last 4 columns of 'Total Revenue' or 'Revenue'.
    Compute 3 consecutive YoY intervals, label them by the ending year, and return a dict.
    Example:
      {
        "YoY2021": 0.25,
        "YoY2022": 0.30,
        "YoY2023": 0.22,
        "AvgGrowth": 0.256
      }
    or None if insufficient data.
    """
    df_fin = fetch_or_load_financials(ticker)
    if df_fin.empty:
        return None

    # Try row labels
    for lbl in ["Total Revenue", "Revenue"]:
        if lbl in df_fin.index:
            rev_series = df_fin.loc[lbl]
            break
    else:
        return None

    # Sort ascending by date
    rev_series = rev_series.sort_index()

    # Need at least 4 data points => 3 yoy intervals
    if len(rev_series) < 4:
        return None

    # Last 4 => final 3 yoy intervals
    rev_series = rev_series.iloc[-4:]

    date_list = list(rev_series.index)  # Timestamps
    yoy_values = []
    result = {}

    for i in range(len(date_list) - 1):
        start_date = date_list[i]
        end_date   = date_list[i+1]
        start_rev  = rev_series.loc[start_date]
        end_rev    = rev_series.loc[end_date]
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
    results = []
    for ticker in MAG7_TICKERS:
        yoy_dict = get_3yr_revenue_growth_with_labels(ticker)
        if yoy_dict is None:
            continue

        row_data = {"Ticker": ticker}
        row_data.update(yoy_dict)
        results.append(row_data)

    if not results:
        print("No valid revenue data for Magnificent 7 tickers.")
        return

    df_res = pd.DataFrame(results)
    df_res.sort_values("AvgGrowth", ascending=False, inplace=True)

    print("=== Magnificent 7 by last 3 consecutive yoy revenue growth ===")
    for idx, row in df_res.iterrows():
        ticker = row["Ticker"]
        # Find yoy columns
        yoy_cols = [c for c in df_res.columns if c.startswith("YoY") and c != "AvgGrowth"]
        yoy_cols = sorted(yoy_cols)  # e.g. ["YoY2020", "YoY2021", "YoY2022"]
        yoy_strs = []
        for ccol in yoy_cols:
            val = row[ccol] * 100  # convert to percent
            yoy_strs.append(f"{ccol}={val:6.1f}%")

        avg_val = row["AvgGrowth"] * 100
        yoy_str = "  ".join(yoy_strs)
        print(f"{ticker:<6}  {yoy_str}  Avg={avg_val:6.1f}%")

    # Optionally save to CSV
    df_res.to_csv("mag7_revgrowth.csv", index=False)
    print("\nSaved mag7_revgrowth.csv. Cached data in 'cache_financials' folder.")

if __name__ == "__main__":
    main()
