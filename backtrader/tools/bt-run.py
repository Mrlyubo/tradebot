import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Select top 7 market cap growth stock each year, and invest it next year. CAGR = 31%, 8 years 745%. Compared to VOO 8.45%, NVDA 44%.
top_stocks_per_year = {2017: ['AMD', 'NVDA', 'FCX', 'TRGP', 'TPL', 'OKE', 'STLD'], 2018: ['ALGN', 'ANET', 'TTWO', 'BA', 'NVDA', 'NVR', 'FSLR'], 2019: ['ENPH', 'DXCM', 'AXON', 'LULU', 'KDP', 'AMD', 'FTNT'], 2020: ['ENPH', 'PODD', 'AMD', 'PAYC', 'LRCX', 'TER', 'BLDR'], 2021: ['TSLA', 'ENPH', 'MRNA', 'CRWD', 'GNRC', 'FCX', 'ALB'], 2022: ['DVN', 'F', 'FANG', 'FTNT', 'NVDA', 'NUE', 'BLDR'], 2023: ['FSLR', 'TPL', 'OXY', 'STLD', 'SMCI', 'ENPH', 'HES'], 2024: ['SMCI', 'NVDA', 'CRWD', 'META', 'PLTR', 'PANW', 'BLDR'], 2025: ['VST', 'PLTR', 'UAL', 'TPL', 'CEG', 'TRGP', 'NVDA']}

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
        price = row['Close']
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
    monthly_df['SMA_3'] = monthly_df['Close'].rolling(window=3, min_periods=1).mean()
    monthly_df['RSI'] = compute_RSI(monthly_df['Close'], window=3)
    monthly_df['BB_Mean'] = monthly_df['Close'].rolling(window=3, min_periods=1).mean()
    monthly_df['BB_std'] = monthly_df['Close'].rolling(window=3, min_periods=1).std()
    monthly_df['BB_lower'] = monthly_df['BB_Mean'] - 2 * monthly_df['BB_std']
    monthly_df['BB_upper'] = monthly_df['BB_Mean'] + 2 * monthly_df['BB_std']
    monthly_df['Vol_Avg'] = monthly_df['Volume'].rolling(window=3, min_periods=1).mean()

    # Baseline volatility from full daily data.
    daily_returns = df['Close'].pct_change().dropna()
    baseline_vol = daily_returns.std()

    cum_shares = 0.0
    portfolio_vals = []
    invested_money = []
    cumulative_invested = 0.0

    # For trailing stop, track the monthly peak.
    monthly_peak = -np.inf

    for date, row in monthly_df.iterrows():
        price = row['Close']
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
        vol_factor = 1.2 if row['Volume'] > row['Vol_Avg'] else 0.9
        overall_multiplier = base_multiplier * rsi_factor * boll_factor * vol_factor

        # Compute monthly volatility using daily data for that month.
        month_str = date.strftime('%Y-%m')
        daily_month = df[df.index.strftime('%Y-%m') == month_str]['Close'].pct_change().dropna()
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
    df = yf.download(mag7_tickers, start=start_date, end=end_date)['Close']
    df.index = pd.to_datetime(df.index)
    normalized = df.div(df.iloc[0])
    etf_series = normalized.mean(axis=1) * start_capital
    etf_df = pd.DataFrame({"Close": etf_series, "Volume": 1}, index=etf_series.index)
    return etf_df

# --- Self-Made dynamic ETF ---
def create_dynamic_etf_df(top_stocks_per_year, start_date, end_date, start_capital=0.0):
    """
    Constructs a dynamic ETF from yearly top stock selections:
    - Downloads daily close prices for all stocks
    - Creates yearly portfolios based on top_stocks_per_year
    - Normalizes each stock to its start-of-year value
    - Takes equally weighted average within each year
    - Scales by the starting capital
    
    Parameters:
    - top_stocks_per_year: dict with years as keys and lists of stock tickers as values
    - start_date: start date for data download
    - end_date: end date for data download
    - start_capital: initial capital to scale the ETF values
    
    Returns:
    - DataFrame with 'Close' and 'Volume' columns
    """
    # Get all unique stocks
    # top_stocks_per_year = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "TSLA", "NVDA"]
    for year in sorted(top_stocks_per_year.keys()):
        df = yf.download(top_stocks_per_year[year], start=start_date, end=end_date)['Close']
    df.index = pd.to_datetime(df.index)
    normalized = df.div(df.iloc[0])
    etf_series = normalized.mean(axis=1) * start_capital
    etf_df = pd.DataFrame({"Close": etf_series, "Volume": 1}, index=etf_series.index)
    return etf_df

def get_stock_data(ticker, start_date, end_date):
    """Download stock data for a given ticker."""
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def dynamic_multi_stock_portfolio_series(
    top_stocks_per_year, start_date, end_date, monthly_investment=1000.0):
    """
    Dynamic multi-stock portfolio strategy:
    - Invests monthly_investment each month
    - Distributes investment equally among selected stocks for each year
    - Rebalances portfolio annually according to top_stocks_per_year
    """
    # Initialize portfolio tracking
    portfolio = pd.DataFrame()
    all_stocks = set()
    for stocks in top_stocks_per_year.values():
        all_stocks.update(stocks)
    
    # Download data for all stocks
    stock_data = {}
    for ticker in all_stocks:
        stock_data[ticker] = get_stock_data(ticker, start_date, end_date)
    
    # Initialize tracking variables
    portfolio_values = []
    invested_amounts = []
    dates = []
    holdings = {ticker: 0 for ticker in all_stocks}  # Track shares held for each stock
    
    print(f"~ year: {year}")
    
    # Process each year
    for year in range(int(start_date[:4]), int(end_date[:4])):
        print(f"~ year: {year}")
        if year not in top_stocks_per_year:
            continue
        
        current_stocks = top_stocks_per_year[year]
        print(f"~current_stocks: {current_stocks}")
        monthly_per_stock = monthly_investment / len(current_stocks)
        print(f"~monthly_per_stock: {monthly_per_stock}")
        
        # Process each month in the year
        for month in range(1, 13):
            current_date = pd.Timestamp(f"{year}-{month:02d}-01")
            if current_date >= pd.Timestamp(end_date):
                break
                
            # Find the first trading day of the month
            for stock in current_stocks:
                df = stock_data[stock]
                monthly_data = df[df.index.year == year]
                if monthly_data.empty:
                    continue
                    
                monthly_data = monthly_data[monthly_data.index.month == month]
                if monthly_data.empty:
                    continue
                    
                first_trading_day = monthly_data.index[0]
                price = monthly_data.loc[first_trading_day, 'Close']
                
                # Buy shares
                new_shares = monthly_per_stock / price
                holdings[stock] += new_shares
            
            # Calculate portfolio value for this date
            total_value = 0
            for stock in all_stocks:
                if holdings[stock] > 0:
                    df = stock_data[stock]
                    if first_trading_day in df.index:
                        price = df.loc[first_trading_day, 'Close']
                        total_value += holdings[stock] * price
            
            portfolio_values.append(total_value)
            invested_amounts.append(monthly_investment * (len(portfolio_values)))
            dates.append(first_trading_day)
            
        # Year-end rebalancing (optional)
        # You could add rebalancing logic here if desired
    
    return pd.DataFrame({
        "portfolio": portfolio_values,
        "invested": invested_amounts
    }, index=dates)


# --- CAGR Calculation Helper ---
def calculate_cagr(profit_pct, years):
    """
    Calculates the Compound Annual Growth Rate (CAGR).
    years: number of years elapsed
    """
    cagr = ((1+profit_pct/100) ** (1/years) - 1) * 100
    # print(f"profit_pct: {profit_pct}, years {years}")
    return cagr

# --- Annual Reporting Function (with CAGR) ---
def print_annual_results(df, ticker, start_capital=100000.0, start_year=None):
    if start_year is None:
        start_year = int(pd.to_datetime(df.index[0]).year)
    print(f"=== {ticker} Annual Results ===")
    
    # Regular DCA
    dca_df = dca_portfolio_series(df, start_capital)
    dca_yearly = dca_df.resample('YE').last()
    print(f"\n {ticker} Regular DCA : ")
    for date, row in dca_yearly.iterrows():
        year = date.year
        portfolio_value = row['portfolio']
        invested = row['invested']
        profit_pct = (portfolio_value - invested) / invested * 100
        dd = annual_max_drawdown(dca_df['portfolio'], year)
        n = year - start_year + 1
        cagr = calculate_cagr(profit_pct, n)
        print(f"{year}: Portfolio = ${portfolio_value:,.2f}, Invested = ${invested:,.2f}, Profit = {profit_pct:.2f}%, CAGR = {cagr:.2f}%, Max Drawdown = {dd:.2f}%")
    
    # Risk-Managed Enhanced Weighted DCA
    rmewdca_df = risk_managed_enhanced_dca_portfolio_series(df, start_capital, stop_loss_pct=20)
    rmewdca_yearly = rmewdca_df.resample('YE').last()
    print(f"\n {ticker} Risk-Managed Enhanced Weighted DCA:")
    for date, row in rmewdca_yearly.iterrows():
        year = date.year
        portfolio_value = row['portfolio']
        invested = row['invested']
        profit_pct = (portfolio_value - invested) / invested * 100
        dd = annual_max_drawdown(rmewdca_df['portfolio'], year)
        n = year - start_year + 1
        cagr = calculate_cagr(profit_pct, n)
        print(f"{year}: Portfolio = ${portfolio_value:,.2f}, Invested = ${invested:,.2f}, Profit = {profit_pct:.2f}%, CAGR = {cagr:.2f}%, Max Drawdown = {dd:.2f}%")
    
    print("\n" + "="*50 + "\n")

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
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    data[name] = df

# Create our self-made ETF ("Mag7 ETF") from the Magnificent 7 stocks.
mag7_etf_df = create_mag7_etf_df(start_date, end_date, start_capital)
data["Mag7 ETF"] = mag7_etf_df

dynamic_etf_df = create_dynamic_etf_df(
            top_stocks_per_year,
            start_date,
            end_date,
            100000
        )

# print(f"~dynamic_etf_df:  {dynamic_etf_df.head(n=50).to_string()}")
data["Top 7 growth ETF"] = dynamic_etf_df


# --- Print Annual Results for Each Asset ---
for name, df in data.items():
    print_annual_results(df, name, start_capital)
