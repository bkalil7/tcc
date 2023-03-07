import pandas as pd

tickers = pd.read_csv("distance_data/ticker_b.csv", header=None)
prices = pd.read_csv("distance_data/Pt.csv", header=None)
returns = pd.read_csv("distance_data/Rt.csv", header=None)
prices.columns = tickers[0]
returns.columns = tickers[0]
prices.to_csv("distance_data/Pt_formatted.csv", index=False)
returns.to_csv("distance_data/Rt_formatted.csv", index=False)