### Step 1: Import Required Libraries

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import coint
from functools import reduce
from itertools import combinations

### Step 2: Fetching and Preprocessing Data


def get_market_data(symbols, start_date, end_date, interval="1d", return_change=False):
    closing_prices = yf.download(
        symbols, start=start_date, end=end_date, interval=interval
    )["Adj Close"]
    closing_prices.columns = closing_prices.columns.str.replace("=F", "").str.replace(
        "DX-Y.NYB", "DXY"
    )
    if return_change:
        return closing_prices.pct_change().dropna()
    return closing_prices.dropna()


### Step 3: Cointegration and Correlation Analysis


def find_cointegrated_pairs(closes):
    n = len(closes.columns)
    pvalue_matrix = np.ones((n, n))
    pairs = []

    for i in range(n):
        S1 = closes.iloc[:, i]
        for j in range(i + 1, n):
            S2 = closes.iloc[:, j]
            score, pvalue, _ = coint(S1, S2)
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((closes.columns[i], closes.columns[j]))

    return pvalue_matrix, pairs


### Step 4: Visualization


def plot_heatmap(pvalue_matrix, columns):
    sns.heatmap(
        pvalue_matrix,
        xticklabels=columns,
        yticklabels=columns,
        cmap="RdYlGn_r",
        mask=(pvalue_matrix >= 0.95),
    )
    plt.show()


def plot_cumulative_returns(closes):
    cumulative_returns = (1 + closes.pct_change()).cumprod()
    cumulative_returns.plot(figsize=(15, 7))
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.show()


def plot_rolling_correlation(data, pair, window=17):
    rolling_corr = (
        data[list(pair)]
        .rolling(window=window)
        .corr()
        .iloc[0::2, -1]
        .reset_index(drop=True)
    )
    rolling_corr.plot(figsize=(15, 7))
    plt.title(f"Rolling Correlation between {pair[0]} and {pair[1]}")
    plt.show()


def plot_spread(series1, series2, beta, alpha):
    spread = np.log(series1) - beta * np.log(series2) - alpha
    spread.plot(figsize=(15, 7))
    plt.axhline(spread.mean(), color="black")
    plt.axhline(spread.mean() + spread.std(), color="red", linestyle="--")
    plt.axhline(spread.mean() - spread.std(), color="green", linestyle="--")
    plt.ylabel("Spread")
    plt.show()


### Step 5: Analyzing Intraday Data


# def get_intraday_data(symbol, interval="5m", period="60d", start_date=None):
#     if start_date:
#         end_date = start_date - datetime.timedelta(days=int(period.replace("d", "")))
#         df = yf.download(symbol, start=end_date, end=start_date, interval=interval)
#     else:
#         df = yf.Ticker(symbol).history(period=period, interval=interval)
#     df.rename(columns={"Close": symbol}, inplace=True)
#     return df[[symbol]]


def get_intraday_data(symbol, interval="5m", period="60d", start_date=None):
    if start_date:
        end_date = start_date - datetime.timedelta(days=int(period.replace("d", "")))
        df = yf.download(symbol, start=end_date, end=start_date, interval=interval)
    else:
        df = yf.Ticker(symbol).history(period=period, interval=interval)

    # Modify column names to remove '=F' or other extensions
    simplified_name = symbol.replace("=F", "").replace("DX-Y.NYB", "DXY")
    df.rename(columns={"Close": simplified_name}, inplace=True)
    return df[[simplified_name]]


def split_periods(data):
    periods = {
        "us_open": data.between_time("07:00:00", "11:00:00").sort_index(),
        "us_mid": data.between_time("11:00:01", "15:00:00").sort_index(),
        "us_close": data.between_time("15:00:01", "17:00:00").sort_index(),
        "us_apac": data.between_time("17:00:01", "02:00:00").sort_index(),
        "us_emea": data.between_time("02:00:01", "07:00:00").sort_index(),
    }
    return periods


### Step 6: Main Execution

if __name__ == "__main__":
    # Parameters
    start_date = "2010-01-01"
    end_date = "2023-12-20"
    symbols = ["ZN=F", "DX-Y.NYB", "CL=F", "GC=F", "NQ=F", "^DJI"]

    # Fetch Data
    closes = get_market_data(symbols, start_date, end_date)
    plot_cumulative_returns(closes)

    # Cointegration Analysis
    pvalue_matrix, coint_pairs = find_cointegrated_pairs(closes)
    print("Cointegrated Pairs:", coint_pairs)
    plot_heatmap(pvalue_matrix, closes.columns)

    # Intraday Data Analysis
    interval = "1m"
    period = "5d"
    start_date = datetime.date(2024, 8, 27)

    # Fetching and Merging Intraday Data
    intraday_data = [
        get_intraday_data(
            symbol, interval=interval, period=period, start_date=start_date
        )
        for symbol in symbols
    ]
    intraday_data = reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True),
        intraday_data,
    )

    # Split Data into Periods
    periods = split_periods(intraday_data)

    # Plot Rolling Correlation for a Specific Pair in a Specific Period
    plot_rolling_correlation(periods["us_mid"], ("ZN", "GC"))
    plot_rolling_correlation(periods["us_mid"], ("ZN", "DXY"))

    # Example: Plot Spread for a Cointegrated Pair
    if coint_pairs:
        symbol1, symbol2 = coint_pairs[0]
        beta = 1.0  # or calculate based on historical regression
        alpha = 0.0  # or calculate based on historical regression
        plot_spread(closes[symbol1], closes[symbol2], beta, alpha)
