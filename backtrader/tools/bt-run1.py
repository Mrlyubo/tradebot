import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time 

# Top 7 stock  market cap groth rate each year, CAGR = 32%, (8 yrs return 830%), maxdrawdown = -13%
# Compared to VOO CAGR 8.45% (8 yrs return 91%), maxdrawdon = -16%


CACHE_DIR = "./Tiingo"
TIINGO_API_KEY = 'e5562fd4c66b35766597200aa8152f8ae0a15e8f'
def download_tiingo_data(tickers, start_date="2017-01-01", end_date="2025-12-31"):
    """
    Downloads historical daily prices from Tiingo for the given ticker(s)
    between start_date and end_date (YYYY-MM-DD format).
    Saves the data as CSV in the CACHE_DIR folder (one file per ticker).
    Returns a combined pandas DataFrame with columns: date, adjClose, etc.,
    and an additional column called 'Ticker' indicating which rows belong to 
    which ticker. If the DataFrame does not contain a 'close' column, it renames 
    'adjClose' to 'close'.

    :param tickers: A single ticker as a string or a list of tickers.
    :param start_date: Start date in 'YYYY-MM-DD' format.
    :param end_date: End date in 'YYYY-MM-DD' format.
    :return: A pandas DataFrame with data for all requested tickers.
    """

    # If a single string is passed in, convert it to a list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    all_dataframes = []

    for ticker in tickers:
        # Path to cached CSV
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
            
            # Save the new data to CSV for caching
            df_ticker.to_csv(csv_path, index=False)
    
        # Add a column to record the ticker
        df_ticker["Ticker"] = ticker

        all_dataframes.append(df_ticker)

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

# def fetch_data_with_cache(
#     ticker, 
#     start, 
#     end, 
#     data_cache_folder="Tiigo2", 
#     sleep_seconds=1
# ):
#     """
#     Checks if the data for `ticker` is in local `data_cache_folder`.
#     If the `.pkl` file exists, load it and return as a DataFrame.
#     If not, download from yfinance (pausing briefly to avoid rate limit),
#     then save to a pickle in `data_cache_folder`.

#     Returns: DataFrame with columns ["Open","High","Low","Close","Volume",...] 
#              indexed by date.
#     """
#     # Ensure cache folder exists
#     if not os.path.exists(data_cache_folder):
#         print("not exist")
#         os.makedirs(data_cache_folder)

#     # For simplicity, we do NOT embed start/end dates in the filename.
#     cache_file = os.path.join(data_cache_folder, f"{ticker}_financials.pkl")

#     # 1) Attempt to load from local .pkl
#     if os.path.isfile(cache_file):
#         print(f"[Cache Hit] Loading {ticker} from {cache_file}")
#         df = pd.read_pickle(cache_file)
#         # Optionally verify that it covers the date range, or just trust it
#         return df
    
#     # 2) Otherwise, download from yfinance
#     print(f"[Cache Miss] Downloading {ticker} from yfinance...")
#     time.sleep(sleep_seconds)  # pause to reduce rate-limit issues
#     df = (ticker, start_date=start, end_date=end)

#     # 3) Save to pickle
#     df.to_pickle(cache_file)
#     print(f"Saved {ticker} data to {cache_file}")
#     return df


top7_by_growth = {
    2017: ['AMD', 'NVDA', 'FCX', 'TRGP', 'TPL', 'OKE', 'STLD'], 
    2018: ['ALGN', 'ANET', 'TTWO', 'BA', 'NVDA', 'NVR', 'FSLR'],
    2019: ['ENPH', 'DXCM', 'AXON', 'LULU', 'KDP', 'AMD', 'FTNT'], 
    2020: ['ENPH', 'PODD', 'AMD', 'PAYC', 'LRCX', 'TER', 'BLDR'], 
    2021: ['TSLA', 'ENPH', 'MRNA', 'CRWD', 'GNRC', 'FCX', 'ALB'], 
    2022: ['DVN', 'F', 'FANG', 'FTNT', 'NVDA', 'NUE', 'BLDR'], 
    2023: ['FSLR', 'TPL', 'OXY', 'STLD', 'SMCI', 'ENPH', 'HES'], 
    2024: ['SMCI', 'NVDA', 'CRWD', 'META', 'PLTR', 'PANW', 'BLDR'], 
    2025: ['VST', 'PLTR', 'UAL', 'TPL', 'CEG', 'TRGP', 'NVDA']
}

# --- Technical Indicator Helper ---
def compute_RSI(series, window=3):
    """Compute RSI for a pandas Series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# --- Regular DCA ---
def dca_portfolio_series(df, start_capital=100000.0):
    """
    Regular DCA: Invest equal amount each month (first trading day).
    Returns a DataFrame with:
      - 'portfolio': monthly portfolio value,
      - 'invested': cumulative money invested.
    """
    df.index = pd.to_datetime(df.index)
    monthly_df = df.resample('MS').first()  # First trading day of each month
    num_months = len(monthly_df)
    monthly_investment = start_capital / num_months
    cum_shares = 0.0
    portfolio_vals = []
    invested_money = []
    cumulative_invested = 0.0
    for date, row in monthly_df.iterrows():
        price = row['close']
        shares = monthly_investment / price
        cum_shares += shares
        cumulative_invested += monthly_investment
        portfolio_vals.append(cum_shares * price)
        invested_money.append(cumulative_invested)
    return pd.DataFrame({"portfolio": portfolio_vals, "invested": invested_money}, index=monthly_df.index)

# --- Risk-Managed Enhanced Weighted DCA ---
def risk_managed_enhanced_dca_portfolio_series(df, start_capital=100000.0, stop_loss_pct=20):
    """
    Risk-Managed Enhanced Weighted DCA:
      - Uses enhanced technical multipliers based on 3-month SMA, RSI, Bollinger Bands, and Volume.
      - Adjusts the monthly investment using a risk factor based on monthly volatility.
      - Implements a simple trailing stop: if the current price drops more than stop_loss_pct%
        from the monthly peak, the effective price is capped.
    """
    df.index = pd.to_datetime(df.index)
    monthly_df = df.resample('MS').first()
    num_months = len(monthly_df)
    base_investment = start_capital / num_months

    # Compute enhanced technical indicators on monthly data.
    monthly_df['SMA_3'] = monthly_df['close'].rolling(window=3, min_periods=1).mean()
    monthly_df['RSI'] = compute_RSI(monthly_df['close'], window=3)
    monthly_df['BB_Mean'] = monthly_df['close'].rolling(window=3, min_periods=1).mean()
    monthly_df['BB_std'] = monthly_df['close'].rolling(window=3, min_periods=1).std()
    monthly_df['BB_lower'] = monthly_df['BB_Mean'] - 2 * monthly_df['BB_std']
    monthly_df['BB_upper'] = monthly_df['BB_Mean'] + 2 * monthly_df['BB_std']
    monthly_df['Vol_Avg'] = monthly_df['volume'].rolling(window=3, min_periods=1).mean()

    # Baseline volatility from full daily data.
    daily_returns = df['close'].pct_change().dropna()
    baseline_vol = daily_returns.std()

    cum_shares = 0.0
    portfolio_vals = []
    invested_money = []
    cumulative_invested = 0.0

    # For trailing stop, track the monthly peak.
    monthly_peak = -np.inf

    for date, row in monthly_df.iterrows():
        price = row['close']
        monthly_peak = max(monthly_peak, price)
        # Cap effective price if the price drops more than stop_loss_pct% from the monthly peak.
        effective_price = max(price, (1 - stop_loss_pct/100) * monthly_peak)

        # Enhanced technical multipliers.
        base_multiplier = row['SMA_3'] / price
        rsi = row['RSI']
        rsi_factor = 1.5 if rsi < 30 else 0.7 if rsi > 70 else 1.0
        if price < row['BB_lower']:
            boll_factor = 1.5
        elif price > row['BB_upper']:
            boll_factor = 0.7
        else:
            boll_factor = 1.0
        vol_factor = 1.2 if row['volume'] > row['Vol_Avg'] else 0.9
        overall_multiplier = base_multiplier * rsi_factor * boll_factor * vol_factor

        # Compute monthly volatility using daily data for that month.
        month_str = date.strftime('%Y-%m')
        daily_month = df[df.index.strftime('%Y-%m') == month_str]['close'].pct_change().dropna()
        monthly_vol = daily_month.std() if not daily_month.empty else baseline_vol
        risk_factor = np.clip(baseline_vol / monthly_vol, 0.5, 1.5)

        monthly_investment = base_investment * overall_multiplier * risk_factor

        shares = monthly_investment / price
        cum_shares += shares
        cumulative_invested += monthly_investment
        portfolio_val = cum_shares * effective_price
        portfolio_vals.append(portfolio_val)
        invested_money.append(cumulative_invested)
    return pd.DataFrame({"portfolio": portfolio_vals, "invested": invested_money}, index=monthly_df.index)

# --- Drawdown Helpers ---
def max_drawdown(series):
    running_max = series.cummax()
    dd = (series - running_max) / running_max * 100
    return dd.min()

def annual_max_drawdown(series, year):
    series_year = series[str(year)]
    if len(series_year) == 0:
        return None
    return max_drawdown(series_year)

# --- Self-Made Mag7 ETF ---
def create_mag7_etf_df(start_date, end_date, start_capital=100000.0):
    """
    Constructs a self-made ETF ("Mag7 ETF") from the Magnificent 7 stocks:
    AAPL, AMZN, GOOG, META, MSFT, TSLA, NVDA.
    Simulated by:
      - Downloading daily close prices for these stocks.
      - Normalizing each to its first value.
      - Taking the equally weighted average.
      - Scaling by the starting capital.
    A constant Volume of 1 is assigned.
    """
    mag7_tickers = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "TSLA", "NVDA"]
    df = download_tiingo_data(mag7_tickers, start_date=start_date, end_date=end_date)['close']
    df.index = pd.to_datetime(df.index)
    normalized = df.div(df.iloc[0])
    etf_series = normalized.mean(axis=1) * start_capital
    etf_df = pd.DataFrame({"Close": etf_series, "Volume": 1}, index=etf_series.index)
    return etf_df

# --- Self-Made dynamic ETF ---
def create_dynamic_etf_df(top7_by_growth, start_date, end_date, start_capital=0.0):
    """
    Constructs a dynamic ETF from yearly top stock selections:
    - Downloads daily close prices for all stocks
    - Creates yearly portfolios based on top7_by_growth
    - Normalizes each stock to its start-of-year value
    - Takes equally weighted average within each year
    - Scales by the starting capital
    
    Parameters:
    - top7_by_growth: dict with years as keys and lists of stock tickers as values
    - start_date: start date for data download
    - end_date: end date for data download
    - start_capital: initial capital to scale the ETF values
    
    Returns:
    - DataFrame with 'close' and 'volume' columns
    """
    # For each year, download data (here the implementation is a simple example)
    for year in sorted(top7_by_growth.keys()):
        df = download_tiingo_data(top7_by_growth[year], start_date=start_date, end_date=end_date)['close']
    df.index = pd.to_datetime(df.index)
    normalized = df.div(df.iloc[0])
    etf_series = normalized.mean(axis=1) * start_capital
    etf_df = pd.DataFrame({"Close": etf_series, "Volume": 1}, index=etf_series.index)
    return etf_df


# --- CAGR Calculation Helper ---
def calculate_cagr(profit_pct, years):
    """
    Calculates the Compound Annual Growth Rate (CAGR).
    years: number of years elapsed
    """
    cagr = ((1+profit_pct/100) ** (1/years) - 1) * 100
    return cagr

# --- Annual Reporting Function (with CAGR) ---
def print_annual_results(
    df, 
    ticker, 
    start_capital=100000.0, 
    start_year=None, 
    file_handle=None
):
    """
    Computes and prints the annual results for a given asset (including DCA 
    and Risk-Managed Enhanced Weighted DCA). If the asset is 'Mag7 ETF', also 
    prints the per-ticker dollar-value breakdown at year-end.
    If the asset is 'Top 7 growth ETF', also prints a *dynamic* breakdown 
    (based on which 7 tickers apply for each year in top7_by_growth).
    """
    output_lines = []
    if start_year is None:
        start_year = int(pd.to_datetime(df.index[0]).year)
    header = f"=== {ticker} Annual Results ==="
    output_lines.append(header)
    
    # ------------------------------------------------------------------------
    # 1) Regular DCA
    # ------------------------------------------------------------------------
    dca_df = dca_portfolio_series(df, start_capital)
    dca_yearly = dca_df.resample('YE').last()
    output_lines.append(f"\n For {ticker}, Regular DCA : ")
    
    # Prepare everything needed for the breakdown if needed:
    # 1) For Mag7 ETF:
    mag7_components = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "TSLA", "NVDA"]
    if ticker == "Mag7 ETF":
        start_idx = df.index[0]
        end_idx   = df.index[-1]
        comp_df = download_tiingo_data(mag7_components, start_date=start_idx, end_date=end_idx)["Close"].dropna(how="all")
        comp_norm = comp_df.div(comp_df.iloc[0])  # day-0 normalization

    # 2) For Top 7 growth ETF:
    elif ticker == "Top 7 growth ETF":
        # We'll need access to your global top7_by_growth (or you can pass it as a function parameter).
        # We'll define a small helper to get the close price at (or right before) a given date
        def get_close_at_or_before(df_stk, date):
            df_sub = df_stk.loc[:date]
            if df_sub.empty:
                return None
            return df_sub.iloc[-1]["Close"]
        
        # We also create a global cache so we don't re-download the same ticker for every year
        # (optional optimization). We'll store in a dict: {ticker: df}
        if not hasattr(print_annual_results, "_ticker_cache"):
            print_annual_results._ticker_cache = {}
        ticker_cache = print_annual_results._ticker_cache

        # We'll define a function that ensures we have a DataFrame for a given stock in the cache:
        def ensure_ticker_data(stock):
            if stock not in ticker_cache:
                dtemp = download_tiingo_data(stock, start_date="2010-01-01", end_date="2030-01-01")  # a broad range
                if isinstance(dtemp.columns, pd.MultiIndex):
                    dtemp.columns = dtemp.columns.get_level_values(0)
                ticker_cache[stock] = dtemp.dropna(subset=["Close"])
            return ticker_cache[stock]
    
    # Loop over each year-end row in DCA
    for date, row in dca_yearly.iterrows():
        year = date.year
        portfolio_value = row['portfolio']
        invested = row['invested']
        profit_pct = (portfolio_value - invested) / invested * 100
        dd = annual_max_drawdown(dca_df['portfolio'], year)
        n = year - start_year + 1
        cagr = calculate_cagr(profit_pct, n)

        line = (f"{year}: Portfolio = ${portfolio_value:,.2f}, "
                f"Invested = ${invested:,.2f}, Profit = {profit_pct:.2f}%, "
                f"CAGR = {cagr:.2f}%, Max Drawdown = {dd:.2f}%")
        output_lines.append(line)

        # ---- Breakdown for Mag7
        if ticker == "Mag7 ETF":
            # Find the closest date <= this "year-end" in comp_norm
            # so we can get normalized prices for that day.
            if date in comp_norm.index:
                row_norm = comp_norm.loc[date]
            else:
                # fallback: last available date on or before `date`
                row_norm = comp_norm.loc[:date].iloc[-1]
            
            # sum of all normalized prices for that day
            sum_norm = row_norm.sum()
            breakdown_lines = ["       Breakdown (Mag7):"]
            for stock in mag7_components:
                val = row_norm.get(stock, np.nan)
                if pd.isna(val):
                    continue
                weight_j = val / sum_norm
                dollar_j = weight_j * portfolio_value
                pct_j    = weight_j * 100
                breakdown_lines.append(
                    f"         {stock}: ${dollar_j:,.2f} ({pct_j:.2f}%)"
                )
            output_lines.extend(breakdown_lines)
        
        # ---- Breakdown for Top 7 Growth
        elif ticker == "Top 7 growth ETF":
            from __main__ import top7_by_growth  # ensure we have the dict
            if year not in top7_by_growth:
                # If your range extends past 2025 or before 2017, you might not have a key.
                output_lines.append(f"       [No top7 data for year={year}]")
            else:
                tickers_for_year = top7_by_growth[year]
                if len(tickers_for_year) == 0:
                    output_lines.append(f"       [No tickers in top7_by_growth for {year}]")
                    continue

                # For each ticker, we compute ratio = close(year-end) / close(year-start)
                year_start = pd.Timestamp(f"{year}-01-01")
                ratios = {}
                for st in tickers_for_year:
                    df_st = ensure_ticker_data(st)  # from the cache
                    c0 = get_close_at_or_before(df_st, year_start)
                    c1 = get_close_at_or_before(df_st, date)
                    if c0 is None or c1 is None or c0 == 0:
                        continue
                    ratios[st] = c1 / c0
                
                if len(ratios) == 0:
                    output_lines.append(f"       [No valid ratio data for {year}]")
                    continue

                sum_ratios = sum(ratios.values())
                breakdown_lines = [f"       Breakdown (Top7 for {year}):"]
                for st, ratio_val in ratios.items():
                    w_st = ratio_val / sum_ratios
                    d_st = w_st * portfolio_value
                    p_st = w_st * 100
                    breakdown_lines.append(
                        f"         {st}: ${d_st:,.2f} ({p_st:.2f}%)"
                    )
                output_lines.extend(breakdown_lines)

    # ------------------------------------------------------------------------
    # 2) Risk-Managed Enhanced Weighted DCA
    # ------------------------------------------------------------------------
    rmewdca_df = risk_managed_enhanced_dca_portfolio_series(df, start_capital, stop_loss_pct=20)
    rmewdca_yearly = rmewdca_df.resample('YE').last()
    output_lines.append(f"\n For {ticker}, Risk-Managed Enhanced Weighted DCA:")
    
    # Loop over each year-end in the RMEW-DCA
    for date, row in rmewdca_yearly.iterrows():
        year = date.year
        portfolio_value = row['portfolio']
        invested = row['invested']
        profit_pct = (portfolio_value - invested) / invested * 100
        dd = annual_max_drawdown(rmewdca_df['portfolio'], year)
        n = year - start_year + 1
        cagr = calculate_cagr(profit_pct, n)

        line = (f"{year}: Portfolio = ${portfolio_value:,.2f}, "
                f"Invested = ${invested:,.2f}, Profit = {profit_pct:.2f}%, "
                f"CAGR = {cagr:.2f}%, Max Drawdown = {dd:.2f}%")
        output_lines.append(line)

        # ---- Breakdown for Mag7
        if ticker == "Mag7 ETF":
            if date in comp_norm.index:
                row_norm = comp_norm.loc[date]
            else:
                row_norm = comp_norm.loc[:date].iloc[-1]
            sum_norm = row_norm.sum()
            breakdown_lines = ["       Breakdown (Mag7):"]
            for stock in mag7_components:
                val = row_norm.get(stock, np.nan)
                if pd.isna(val):
                    continue
                weight_j = val / sum_norm
                dollar_j = weight_j * portfolio_value
                pct_j    = weight_j * 100
                breakdown_lines.append(
                    f"         {stock}: ${dollar_j:,.2f} ({pct_j:.2f}%)"
                )
            output_lines.extend(breakdown_lines)

        # ---- Breakdown for Top 7 Growth
        elif ticker == "Top 7 growth ETF":
            from __main__ import top7_by_growth
            if year not in top7_by_growth:
                output_lines.append(f"       [No top7 data for year={year}]")
            else:
                tickers_for_year = top7_by_growth[year]
                if len(tickers_for_year) == 0:
                    output_lines.append(f"       [No tickers in top7_by_growth for {year}]")
                    continue
                
                year_start = pd.Timestamp(f"{year}-01-01")
                # same ratio logic as above
                ratios = {}
                for st in tickers_for_year:
                    df_st = ensure_ticker_data(st)
                    c0 = get_close_at_or_before(df_st, year_start)
                    c1 = get_close_at_or_before(df_st, date)
                    if c0 is None or c1 is None or c0 == 0:
                        continue
                    ratios[st] = c1 / c0
                
                if len(ratios) == 0:
                    output_lines.append(f"       [No valid ratio data for {year}]")
                    continue
                
                sum_ratios = sum(ratios.values())
                breakdown_lines = [f"       Breakdown (Top7 for {year}):"]
                for st, ratio_val in ratios.items():
                    w_st = ratio_val / sum_ratios
                    d_st = w_st * portfolio_value
                    p_st = w_st * 100
                    breakdown_lines.append(
                        f"         {st}: ${d_st:,.2f} ({p_st:.2f}%)"
                    )
                output_lines.extend(breakdown_lines)
    
    output_lines.append("\n" + "="*50 + "\n")

    # --- Print to console ---
    for line in output_lines:
        print(line)

    # --- Optionally write to file ---
    if file_handle is not None:
        file_handle.write("\n".join(output_lines) + "\n")




# --- Settings and Data Download ---
start_date = '2017-01-01'
end_date = '2025-01-01'
start_capital = 100000.0

# Define tickers for our assets: AMZN, NVDA, VOO, SPXL
tickers = {
    "AMZN": "AMZN",
    "NVDA": "NVDA",
    "VOO": "VOO",    # S&P500 ETF
    "SPXL": "SPXL"   # 3x leveraged S&P500 ETF
}

data = {}
for name, ticker in tickers.items():
    df = download_tiingo_data(ticker, start_date=start_date, end_date=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    data[name] = df

# # Create our self-made ETF ("Mag7 ETF") from the Magnificent 7 stocks.
# mag7_etf_df = create_mag7_etf_df(start_date, end_date, start_capital)
# data["Mag7 ETF"] = mag7_etf_df

# dynamic_etf_df = create_dynamic_etf_df(
#     top7_by_growth,
#     start_date,
#     end_date,
#     100000
# )
# data["Top 7 growth ETF"] = dynamic_etf_df

# Open a file to write the annual results locally.
with open("annual_results.txt", "w") as f:
    # --- Print Annual Results for Each Asset ---
    for name, df in data.items():
        print_annual_results(df, name, start_capital, file_handle=f)