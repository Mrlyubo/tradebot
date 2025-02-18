# import backtrader as bt
# import yfinance as yf
# from datetime import datetime, timedelta
# import pandas as pd
# import numpy as np

# class MonthlyInvestStrategy(bt.Strategy):
#     params = (
#         ('monthly_investment', 1000),  # Amount to invest each month
#     )
    
#     def __init__(self):
#         self.orders = {}  # Keep track of orders
#         self.investment_dates = []  # Track investment dates
#         self.monthly_stocks = {}  # Dictionary to store current month's stocks
        
#     def next(self):
#         # Check if it's time for monthly investment
#         current_date = self.data0.datetime.date(0)
        
#         # If it's the first day of the month and we haven't invested yet this month
#         if current_date.day == 1 and current_date not in self.investment_dates:
#             self.investment_dates.append(current_date)
            
#             # Get the year and corresponding stocks
#             year = current_date.year
#             if year in top_stocks_per_year:
#                 current_stocks = top_stocks_per_year[year]
                
#                 # Calculate amount per stock
#                 amount_per_stock = self.params.monthly_investment / len(current_stocks)
                
#                 # Place orders for each stock
#                 for stock in current_stocks:
#                     if stock in self.dnames:  # Check if we have data for this stock
#                         data = getattr(self, stock)
#                         if data._name not in self.orders:
#                             self.orders[data._name] = []
                            
#                         # Calculate shares to buy based on current price
#                         price = data.close[0]
#                         shares = amount_per_stock / price
                        
#                         # Place the buy order
#                         order = self.buy(data=data, size=shares)
#                         self.orders[data._name].append(order)

# def prepare_data(symbols, start_date, end_date):
#     data_frames = {}
#     for symbol in symbols:
#         try:
#             # Download data
#             df = yf.download(symbol, start=start_date, end=end_date)
#             if not df.empty:
#                 # Ensure column names are correct for backtrader
#                 df.columns = [col.lower() for col in df.columns]
#                 # Ensure the index is datetime
#                 df.index = pd.to_datetime(df.index)
#                 data_frames[symbol] = df
#         except Exception as e:
#             print(f"Error downloading {symbol}: {e}")
#     return data_frames

# def run_backtest(data_frames, initial_cash=10000):
#     # Create a cerebro instance
#     cerebro = bt.Cerebro()
    
#     # Add strategy
#     cerebro.addstrategy(MonthlyInvestStrategy)
    
#     # Add data feeds
#     for symbol, df in data_frames.items():
#         data = bt.feeds.PandasData(
#             dataname=df,
#             name=symbol,
#             datetime=None,  # Index is already datetime
#             open='open',
#             high='high',
#             low='low',
#             close='close',
#             volume='volume',
#             openinterest=-1  # Not available from Yahoo
#         )
#         cerebro.adddata(data)
    
#     # Set initial cash
#     cerebro.broker.setcash(initial_cash)
    
#     # Set commission - 0.1% per trade
#     cerebro.broker.setcommission(commission=0.001)
    
#     # Run the backtest
#     initial_portfolio_value = cerebro.broker.getvalue()
#     results = cerebro.run()
#     final_portfolio_value = cerebro.broker.getvalue()
    
#     return initial_portfolio_value, final_portfolio_value

# # Dictionary containing the top stocks per year (as provided)
# top_stocks_per_year = {
#     2017: ['AMD', 'NVDA', 'FCX', 'TRGP', 'TPL', 'OKE', 'STLD']
#     # 2018: ['ALGN', 'ANET', 'TTWO', 'BA', 'NVDA', 'NVR', 'FSLR'],
#     # 2019: ['ENPH', 'DXCM', 'AXON', 'LULU', 'KDP', 'AMD', 'FTNT'],
#     # 2020: ['ENPH', 'PODD', 'AMD', 'PAYC', 'LRCX', 'TER', 'BLDR'],
#     # 2021: ['TSLA', 'ENPH', 'MRNA', 'CRWD', 'GNRC', 'FCX', 'ALB'],
#     # 2022: ['DVN', 'F', 'FANG', 'FTNT', 'NVDA', 'NUE', 'BLDR'],
#     # 2023: ['FSLR', 'TPL', 'OXY', 'STLD', 'SMCI', 'ENPH', 'HES']
# }

# # Get unique symbols from all years
# all_symbols = set()
# for year_stocks in top_stocks_per_year.values():
#     all_symbols.update(year_stocks)

# # Set date range (using only historical data up to 2023)
# start_date = '2017-01-01'
# end_date = '2018-12-31'

# # Prepare data
# print("Downloading data...")
# data_frames = prepare_data(all_symbols, start_date, end_date)

# # Run backtest
# print("Running backtest...")
# initial_value, final_value = run_backtest(data_frames)

# # Calculate and print results
# total_months = len(pd.date_range(start=start_date, end=end_date, freq='M'))
# total_invested = total_months * 1000
# total_return = ((final_value - initial_value) / total_invested) * 100

# print("\nResults:")
# print(f"Initial Portfolio Value: ${initial_value:,.2f}")
# print(f"Final Portfolio Value: ${final_value:,.2f}")
# print(f"Total Invested: ${total_invested:,.2f}")
# print(f"Total Return: {total_return:.2f}%")
# print(f"Absolute Return: ${final_value - initial_value - total_invested:,.2f}")