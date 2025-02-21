import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time
import requests

# Top 7 stock  market cap groth rate each year, CAGR = 32%, (8 yrs return 830%), maxdrawdown = -13%
# Compared to VOO CAGR 8.45% (8 yrs return 91%), maxdrawdon = -16%

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
    df.index = pd.to_datetime(df['date'])
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
    df.index = pd.to_datetime(df['date'])
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

def dynamic_top7_portfolio_details(top7_by_growth, 
                                   start_date="2017-01-01", 
                                   end_date="2025-01-01", 
                                   monthly_investment=1000.0):
    """
    Build a *real* multi-ticker portfolio for 'Top 7 growth',
    buying each month's first trading day in the chosen tickers
    for that calendar year.

    Returns:
      holdings_history: a dictionary keyed by date, each value is another
        dict { 'portfolio_value': float,
               'invested': float,
               'breakdown': {
                   ticker: {
                     'shares': float,
                     'cost':   float,
                     'price':  float,  # price on this day
                     'value':  float
                   }, ...
                }
             }
      Also returns a Pandas DataFrame with monthly rows (same data flattened).
    """
    # 1) Collect all unique tickers
    all_tickers = set()
    for year_tickers in top7_by_growth.values():
        all_tickers.update(year_tickers)

    # 2) Download data once for all tickers (daily)
    all_data = {}
    for tkr in all_tickers:
        df_t = download_tiingo_data(tkr, start_date=start_date, end_date=end_date)
        if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)
        all_data[tkr] = df_t.dropna(subset=["Close"])
    
    # 3) For each ticker, track total SHARES owned, total COST basis
    shares_held = {t: 0.0 for t in all_tickers}
    cost_held   = {t: 0.0 for t in all_tickers}
    
    # This dictionary will map each monthly date -> details
    holdings_history = {}
    
    # We iterate month by month from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")  # every 1st of month
    for dt in date_range:
        current_year = dt.year
        if current_year not in top7_by_growth:
            # If we have no selection for that year, skip buying
            pass
        else:
            tickers_for_year = top7_by_growth[current_year]
            # We'll buy equally among these tickers (if any)
            if len(tickers_for_year) > 0:
                invest_per_ticker = monthly_investment / len(tickers_for_year)
                
                # For each ticker, find the first trading day price
                for tkr in tickers_for_year:
                    df_t = all_data[tkr]
                    # Find the row for dt or the next available trading day
                    # (Because dt might be a weekend/holiday.)
                    df_month = df_t.loc[str(dt.date()):]  # all data from dt onward
                    if df_month.empty:
                        continue
                    first_trading_day = df_month.index[0]
                    first_day_price = df_month.loc[first_trading_day, "Close"]
                    
                    # Buy shares
                    new_shares = invest_per_ticker / first_day_price
                    shares_held[tkr] += new_shares
                    cost_held[tkr]   += invest_per_ticker
        
        # Now let's record the *end-of-month* portfolio value 
        # (or you could do "first trading day" - your choice).
        # For consistency, let's do end-of-month:
        end_of_month = (dt + pd.offsets.MonthEnd(1))  # last calendar day of that month
        if end_of_month > pd.Timestamp(end_date):
            end_of_month = pd.Timestamp(end_date)
        
        # For each ticker, get the price on or before 'end_of_month'
        monthly_breakdown = {}
        total_value = 0.0
        total_invested = 0.0
        
        for tkr in all_tickers:
            if shares_held[tkr] == 0:
                continue
            df_t = all_data[tkr]
            # find price at or before end_of_month
            df_sub = df_t.loc[:end_of_month]
            if df_sub.empty:
                continue
            price_eom = df_sub.iloc[-1]["Close"]  # last known price
            val_eom = shares_held[tkr] * price_eom
            total_value += val_eom
            total_invested += cost_held[tkr]
            monthly_breakdown[tkr] = {
                "shares": shares_held[tkr],
                "cost":   cost_held[tkr],
                "price":  price_eom,
                "value":  val_eom
            }
        
        holdings_history[end_of_month] = {
            "portfolio_value": total_value,
            "invested":        total_invested,
            "breakdown":       monthly_breakdown
        }
    
    # Convert holdings_history to a monthly DataFrame (optional)
    rows = []
    for dt, info in holdings_history.items():
        row = {
            "date": dt,
            "portfolio_value": info["portfolio_value"],
            "invested":        info["invested"]
        }
        rows.append(row)
    hist_df = pd.DataFrame(rows).set_index("date").sort_index()
    
    return holdings_history, hist_df


def print_annual_results(
    df, 
    ticker, 
    start_capital=100000.0, 
    start_year=None, 
    file_handle=None
):
    """
    Computes and prints the annual results for a given asset (including DCA 
    and Risk-Managed Enhanced Weighted DCA). 
    Also prints a breakdown if the asset is 'Mag7 ETF' or 'Top 7 growth ETF'.
    
    For 'Top 7 growth ETF', we rely on a *real multi-ticker portfolio* approach
    (monthly invests in whichever top7 are valid that year). This is done by
    calling `dynamic_top7_portfolio_details()` under the hood, so we can see
    ticker-level shares, cost, and year-end value.
    """
    output_lines = []
    if start_year is None:
        start_year = int(pd.to_datetime(df['date']).year)
    header = f"=== {ticker} Annual Results ==="
    output_lines.append(header)

    # --------------------------------------------------------------------
    # 1) Regular DCA - (for single ticker or for a synthetic timeseries)
    # --------------------------------------------------------------------
    dca_df = dca_portfolio_series(df, start_capital)
    dca_yearly = dca_df.resample('YE').last()
    output_lines.append(f"\n For {ticker}, Regular DCA : ")
    # Prepare breakdown logic if "Mag7 ETF"
    mag7_components = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "TSLA", "NVDA"]
    if ticker == "Mag7 ETF":
        start_idx = df.index[0]
        end_idx   = df.index[-1]
        comp_df = download_tiingo_data(mag7_components,
                start_date==start_idx, end_date=end_idx)["Close"].dropna(how="all")
        comp_norm = comp_df.div(comp_df.iloc[0])  # day-0 normalization

    # If "Top 7 growth ETF", build a real portfolio with monthly invests
    elif ticker == "Top 7 growth ETF":
        # We'll do it once, store in memory, so we can pick out the year-end breakdown
        total_months = (pd.Period(end_date, freq='M') - pd.Period(start_date, freq='M')) + 1
        holdings_history, hist_df = dynamic_top7_portfolio_details(
            top7_by_growth=top7_by_growth,
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
            monthly_investment=(start_capital / total_months)  # or your choice
        )
    # We'll later look up each year's 12/31 in `holdings_history`.

    # Loop over each year-end in the DCA results
    for date, row in dca_yearly.iterrows():
        year = date.year
        portfolio_value = row['portfolio']
        invested = row['invested']
        profit_pct = (portfolio_value - invested) / invested * 100 if invested != 0 else 0
        dd = annual_max_drawdown(dca_df['portfolio'], year)
        n = year - start_year + 1
        cagr = calculate_cagr(profit_pct, n)

        line = (f"{year}: Portfolio = ${portfolio_value:,.2f}, "
                f"Invested = ${invested:,.2f}, Profit = {profit_pct:.2f}%, "
                f"CAGR = {cagr:.2f}%, Max Drawdown = {dd:.2f}%")
        output_lines.append(line)

        # ----- Breakdown for Mag7
        if ticker == "Mag7 ETF":
            if date in comp_norm.index:
                row_norm = comp_norm.loc[date]
            else:
                # fallback: last available date on or before `date`
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

        # ----- Breakdown for Top 7 growth
        elif ticker == "Top 7 growth ETF":
            # We'll look up the last monthly entry in holdings_history for this year
            # e.g. 12/31 or last trading day in December
            year_end_str = f"{year}-12-31"
            # Find the nearest date in holdings_history <= year_end_str
            all_dates = sorted(list(holdings_history.keys()))
            cutoff_dates = [d for d in all_dates if d <= pd.Timestamp(year_end_str)]
            if not cutoff_dates:
                output_lines.append(f"       [No holdings data up to {year_end_str}]")
            else:
                chosen_date = cutoff_dates[-1]  # last date in or before 12/31
                info = holdings_history[chosen_date]  # { 'portfolio_value', 'invested', 'breakdown': {ticker: {...}} }
                # Now we can show each ticker’s shares, cost, year-end price, year-end value
                breakdown_lines = [f"       Breakdown (Top7 for {year}, as of {chosen_date.date()}):"]
                total_port_val = info["portfolio_value"]
                for st, st_info in info["breakdown"].items():
                    sh   = st_info["shares"]
                    cost = st_info["cost"]
                    px   = st_info["price"]
                    val  = st_info["value"]
                    if total_port_val > 0:
                        pct  = (val / total_port_val) * 100
                    else:
                        pct = 0
                    breakdown_lines.append(
                        f"         {st}: shares={sh:,.3f}, cost=${cost:,.2f}, price=${px:,.2f}, "
                        f"value=${val:,.2f}, {pct:.2f}% of port"
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

        # Mag7 breakdown
        if ticker == "Mag7 ETF":
            # same approach as above
            # ...
            pass

        # Top 7 breakdown
        elif ticker == "Top 7 growth ETF":
            year_end_str = f"{year}-12-31"
            all_dates = sorted(list(holdings_history.keys()))
            cutoff_dates = [d for d in all_dates if d <= pd.Timestamp(year_end_str)]
            if not cutoff_dates:
                output_lines.append(f"       [No holdings data up to {year_end_str}]")
            else:
                chosen_date = cutoff_dates[-1]
                info = holdings_history[chosen_date]
                breakdown_lines = [f"       Breakdown (Top7 for {year}, as of {chosen_date.date()}):"]
                total_port_val = info["portfolio_value"]
                for st, st_info in info["breakdown"].items():
                    sh   = st_info["shares"]
                    cost = st_info["cost"]
                    px   = st_info["price"]
                    val  = st_info["value"]
                    pct  = (val / total_port_val)*100 if total_port_val>0 else 0
                    breakdown_lines.append(
                        f"         {st}: shares={sh:,.3f}, cost=${cost:,.2f}, price=${px:,.2f}, "
                        f"value=${val:,.2f}, {pct:.2f}% of port"
                    )
                output_lines.extend(breakdown_lines)

    output_lines.append("\n" + "="*50 + "\n")

    # --- Print to console ---
    for line in output_lines:
        print(line)

    # --- Optionally write to file ---
    if file_handle is not None:
        file_handle.write("\n".join(output_lines) + "\n")


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
    and Risk-Managed Enhanced Weighted DCA). 
    Also prints a breakdown if the asset is 'Mag7 ETF' or 'Top 7 growth ETF'.
    
    For 'Top 7 growth ETF', we rely on a *real multi-ticker portfolio* approach
    (monthly invests in whichever top7 are valid that year). This is done by
    calling `dynamic_top7_portfolio_details()` under the hood, so we can see
    ticker-level shares, cost, and year-end value.
    """
    
    output_lines = []
    if start_year is None:
        start_year = int(pd.to_datetime(df.index[0]).year)
    header = f"=== {ticker} Annual Results ==="
    output_lines.append(header)

    # --------------------------------------------------------------------
    # 1) Regular DCA - (for single ticker or for a synthetic timeseries)
    # --------------------------------------------------------------------
    dca_df = dca_portfolio_series(df, start_capital)
    dca_yearly = dca_df.resample('YE').last()
    output_lines.append(f"\n For {ticker}, Regular DCA : ")

    # Prepare breakdown logic if "Mag7 ETF"
    mag7_components = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "TSLA", "NVDA"]
    if ticker == "Mag7 ETF":
        start_idx = df.index[0]
        end_idx   = df.index[-1]
        comp_df = download_tiingo_data(mag7_components, start=start_idx, end=end_idx)["Close"].dropna(how="all")
        comp_norm = comp_df.div(comp_df.iloc[0])  # day-0 normalization

    # If "Top 7 growth ETF", build a real portfolio with monthly invests
    elif ticker == "Top 7 growth ETF":
        # We'll do it once, store in memory, so we can pick out the year-end breakdown
        total_months = (pd.Period(end_date, freq='M') - pd.Period(start_date, freq='M')) + 1
        holdings_history, hist_df = dynamic_top7_portfolio_details(
            top7_by_growth=top7_by_growth,
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
            monthly_investment=(start_capital / total_months)  # or your choice
        )
        # We'll later look up each year's 12/31 in `holdings_history`.

    # Loop over each year-end in the DCA results
    for date, row in dca_yearly.iterrows():
        year = date.year
        portfolio_value = row['portfolio']
        invested = row['invested']
        profit_pct = (portfolio_value - invested) / invested * 100 if invested != 0 else 0
        dd = annual_max_drawdown(dca_df['portfolio'], year)
        n = year - start_year + 1
        cagr = calculate_cagr(profit_pct, n)

        line = (f"{year}: Portfolio = ${portfolio_value:,.2f}, "
                f"Invested = ${invested:,.2f}, Profit = {profit_pct:.2f}%, "
                f"CAGR = {cagr:.2f}%, Max Drawdown = {dd:.2f}%")
        output_lines.append(line)

        # ----- Breakdown for Mag7
        if ticker == "Mag7 ETF":
            if date in comp_norm.index:
                row_norm = comp_norm.loc[date]
            else:
                # fallback: last available date on or before `date`
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

        # ----- Breakdown for Top 7 growth
        elif ticker == "Top 7 growth ETF":
            # We'll look up the last monthly entry in holdings_history for this year
            # e.g. 12/31 or last trading day in December
            year_end_str = f"{year}-12-31"
            # Find the nearest date in holdings_history <= year_end_str
            all_dates = sorted(list(holdings_history.keys()))
            cutoff_dates = [d for d in all_dates if d <= pd.Timestamp(year_end_str)]
            if not cutoff_dates:
                output_lines.append(f"       [No holdings data up to {year_end_str}]")
            else:
                chosen_date = cutoff_dates[-1]  # last date in or before 12/31
                info = holdings_history[chosen_date]  # { 'portfolio_value', 'invested', 'breakdown': {ticker: {...}} }
                # Now we can show each ticker’s shares, cost, year-end price, year-end value
                breakdown_lines = [f"       Breakdown (Top7 for {year}, as of {chosen_date.date()}):"]
                total_port_val = info["portfolio_value"]
                for st, st_info in info["breakdown"].items():
                    sh   = st_info["shares"]
                    cost = st_info["cost"]
                    px   = st_info["price"]
                    val  = st_info["value"]
                    if total_port_val > 0:
                        pct  = (val / total_port_val) * 100
                    else:
                        pct = 0
                    breakdown_lines.append(
                        f"         {st}: shares={sh:,.3f}, cost=${cost:,.2f}, price=${px:,.2f}, "
                        f"value=${val:,.2f}, {pct:.2f}% of port"
                    )
                output_lines.extend(breakdown_lines)

    # ------------------------------------------------------------------------
    # 2) Risk-Managed Enhanced Weighted DCA
    # ------------------------------------------------------------------------
    rmewdca_df = risk_managed_enhanced_dca_portfolio_series(df, start_capital, stop_loss_pct=20)
    rmewdca_yearly = rmewdca_df.resample('YE').last()
    output_lines.append(f"\n For {ticker}, Risk-Managed Enhanced Weighted DCA:")
    
    for date, row in rmewdca_yearly.iterrows():
        year = date.year
        portfolio_value = row['portfolio']
        invested = row['invested']
        profit_pct = (portfolio_value - invested) / invested * 100 if invested != 0 else 0
        dd = annual_max_drawdown(rmewdca_df['portfolio'], year)
        n = year - start_year + 1
        cagr = calculate_cagr(profit_pct, n)

        line = (f"{year}: Portfolio = ${portfolio_value:,.2f}, "
                f"Invested = ${invested:,.2f}, Profit = {profit_pct:.2f}%, "
                f"CAGR = {cagr:.2f}%, Max Drawdown = {dd:.2f}%")
        output_lines.append(line)

        # Mag7 breakdown
        if ticker == "Mag7 ETF":
            # same approach as above
            # ...
            pass

        # Top 7 breakdown
        elif ticker == "Top 7 growth ETF":
            year_end_str = f"{year}-12-31"
            all_dates = sorted(list(holdings_history.keys()))
            cutoff_dates = [d for d in all_dates if d <= pd.Timestamp(year_end_str)]
            if not cutoff_dates:
                output_lines.append(f"       [No holdings data up to {year_end_str}]")
            else:
                chosen_date = cutoff_dates[-1]
                info = holdings_history[chosen_date]
                breakdown_lines = [f"       Breakdown (Top7 for {year}, as of {chosen_date.date()}):"]
                total_port_val = info["portfolio_value"]
                for st, st_info in info["breakdown"].items():
                    sh   = st_info["shares"]
                    cost = st_info["cost"]
                    px   = st_info["price"]
                    val  = st_info["value"]
                    pct  = (val / total_port_val)*100 if total_port_val>0 else 0
                    breakdown_lines.append(
                        f"         {st}: shares={sh:,.3f}, cost=${cost:,.2f}, price=${px:,.2f}, "
                        f"value=${val:,.2f}, {pct:.2f}% of port"
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
    # "NVDA": "NVDA",
    # "VOO": "VOO",    # S&P500 ETF
    # "SPXL": "SPXL"   # 3x leveraged S&P500 ETF
}

data = {}
for name, ticker in tickers.items():
    df = download_tiingo_data(ticker, start_date=start_date, end_date=end_date)
    print(f"df head: {df.head()}")
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
