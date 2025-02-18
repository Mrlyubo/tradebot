import pandas as pd
import os 

def find_top_market_caps(df, top_n=7):
    """
    Given a DataFrame with columns [Ticker, Company, Year, MarketCap],
    returns a new DataFrame with the top N market cap companies for each year.
    """
    # Group by Year, then sort within each group by MarketCap descending, pick top N
    df_top = (
        df.sort_values(["Year", "MarketCap"], ascending=[True, False])
          .groupby("Year")
          .head(top_n)
          .reset_index(drop=True)
    )
    return df_top

def compute_yoy_growth(df):
    """
    Given a DataFrame with columns [Ticker, Year, MarketCap],
    computes year-over-year percentage growth in MarketCap for each Ticker.
    Returns a new DataFrame with an additional 'YoYGrowth' column.
    
    YoYGrowth (%) = ((MarketCap_current - MarketCap_previous) / MarketCap_previous) * 100
    """
    # We assume each (Ticker, Year) is unique, but a Ticker can appear across multiple years.
    
    # Sort by Ticker, then by Year, so we can compute shift(1) for MarketCap
    df_sorted = df.sort_values(["Ticker", "Year"]).copy()
    
    # Group by Ticker, then compute previous year's MarketCap
    df_sorted["PrevMarketCap"] = df_sorted.groupby("Ticker")["MarketCap"].shift(1)
    
    # Compute YoY Growth, handle cases where PrevMarketCap is missing or zero
    df_sorted["YoYGrowth"] = ((df_sorted["MarketCap"] - df_sorted["PrevMarketCap"]) 
                              / df_sorted["PrevMarketCap"] * 100)
    
    # If PrevMarketCap was NaN or 0, YoYGrowth might be NaN or inf
    # We can choose to fill or keep them as is. Here we'll keep them as NaN.
    
    return df_sorted

def find_top_growth(df, top_n=7):
    """
    Given a DataFrame that includes a 'YoYGrowth' column,
    returns a new DataFrame with the top N growth companies for each year (by percentage).
    """
    # We only consider rows where YoYGrowth is valid
    df_valid = df.dropna(subset=["YoYGrowth"]).copy()
    
    # Sort by Year ascending, then YoYGrowth descending
    df_top_growth = (
        df_valid.sort_values(["Year", "YoYGrowth"], ascending=[True, False])
               .groupby("Year")
               .head(top_n)
               .reset_index(drop=True)
    )
    return df_top_growth

def main():
    # 1) Load the data from CSV
    here = os.getcwd()
    csv_file = here + "/../sp500/sp500_year_end_marketcaps.csv"  # path to your CSV
    df = pd.read_csv(csv_file)
    
    # Ensure correct data types (Year as int, MarketCap as float, etc.)
    df["Year"] = df["Year"].astype(int)
    #df["MarketCap"] = df["MarketCap"].apply(parse_weird_marketcap)
    print(df.head(10))
    df.to_csv("sp500_year_end_marketcaps.csv", index=False)
    
    # 2) Find top 7 by MarketCap each year
    df_top_by_cap = find_top_market_caps(df, top_n=7)
    
    # 3) Compute year-over-year growth for each Ticker
    df_with_growth = compute_yoy_growth(df)
    
    # 4) Find top 7 by YoYGrowth each year
    df_top_growth = find_top_growth(df_with_growth, top_n=7)
    
    # 5) (Optional) Output or print results
    print("=== Top 7 by Market Cap each year ===")
    print(df_top_by_cap)
    
    print("\n=== Top 7 by Market Cap Growth Rate each year ===")
    print(df_top_growth)
    
    # If you want to save them to CSV, do:
    df_top_by_cap.to_csv("top7_by_marketcap.csv", index=False)
    df_top_growth.to_csv("top7_by_growth.csv", index=False)
    print("\nSaved top7_by_marketcap.csv and top7_by_growth.csv")

if __name__ == "__main__":
    main()
