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
import os
import hashlib
import time


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
# def plot_heatmap(pvalue_matrix, columns):
#     sns.heatmap(
#         pvalue_matrix,
#         xticklabels=columns,
#         yticklabels=columns,
#         cmap="RdYlGn_r",
#         mask=(pvalue_matrix >= 0.95),
#     )
#     plt.show()


# def plot_cumulative_returns(closes):
#     cumulative_returns = (1 + closes.pct_change()).cumprod()
#     cumulative_returns.plot(figsize=(15, 7))
#     plt.ylabel("Cumulative Return")
#     plt.legend()
#     plt.show()


# def plot_rolling_correlation(data, pair, window=17):
#     rolling_corr = (
#         data[list(pair)]
#         .rolling(window=window)
#         .corr()
#         .iloc[0::2, -1]
#         .reset_index(drop=True)
#     )
#     rolling_corr.plot(figsize=(15, 7))
#     plt.title(f"Rolling Correlation between {pair[0]} and {pair[1]}")
#     plt.show()


# def plot_spread(series1, series2, beta, alpha):
#     spread = np.log(series1) - beta * np.log(series2) - alpha
#     spread.plot(figsize=(15, 7))
#     plt.axhline(spread.mean(), color="black")
#     plt.axhline(spread.mean() + spread.std(), color="red", linestyle="--")
#     plt.axhline(spread.mean() - spread.std(), color="green", linestyle="--")
#     plt.ylabel("Spread")
#     plt.show()


### Step 5: Analyzing Intraday Data


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


# Create output folder if it doesn't exist
output_folder = "output_cointegration"
os.makedirs(output_folder, exist_ok=True)


# Context manager for saving outputs
class OutputSaver:
    def __init__(self):
        self.run_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.run_folder = os.path.join(output_folder, f"run_{self.run_hash}")
        os.makedirs(self.run_folder, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def save_figure(self, fig, filename):
        fig.savefig(os.path.join(self.run_folder, filename))
        plt.close(fig)

    def save_dataframe(self, df, filename):
        df.to_csv(os.path.join(self.run_folder, filename))

    def save_text(self, text, filename):
        with open(os.path.join(self.run_folder, filename), "w") as f:
            f.write(text)


# Rest of the code remains the same until the visualization functions


def plot_heatmap(pvalue_matrix, columns, saver):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        pvalue_matrix,
        xticklabels=columns,
        yticklabels=columns,
        cmap="RdYlGn_r",
        mask=(pvalue_matrix >= 0.95),
        ax=ax,
    )
    saver.save_figure(fig, "heatmap.png")


def plot_cumulative_returns(closes, saver):
    cumulative_returns = (1 + closes.pct_change()).cumprod()
    fig, ax = plt.subplots(figsize=(15, 7))
    cumulative_returns.plot(ax=ax)
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    saver.save_figure(fig, "cumulative_returns.png")


def plot_rolling_correlation(data, pair, saver, window=17):
    rolling_corr = (
        data[list(pair)]
        .rolling(window=window)
        .corr()
        .iloc[0::2, -1]
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(15, 7))
    rolling_corr.plot(ax=ax)
    ax.set_title(f"Rolling Correlation between {pair[0]} and {pair[1]}")
    saver.save_figure(fig, f"rolling_correlation_{pair[0]}_{pair[1]}.png")


def plot_spread(series1, series2, beta, alpha, saver):
    spread = np.log(series1) - beta * np.log(series2) - alpha
    fig, ax = plt.subplots(figsize=(15, 7))
    spread.plot(ax=ax)
    ax.axhline(spread.mean(), color="black")
    ax.axhline(spread.mean() + spread.std(), color="red", linestyle="--")
    ax.axhline(spread.mean() - spread.std(), color="green", linestyle="--")
    ax.set_ylabel("Spread")
    saver.save_figure(fig, f"spread_{series1.name}_{series2.name}.png")


# Main execution
if __name__ == "__main__":
    with OutputSaver() as saver:
        # Parameters
        start_date = "2010-01-01"
        end_date = "2023-12-20"
        symbols = ["ZN=F", "DX-Y.NYB", "CL=F", "GC=F", "NQ=F", "^DJI"]

        # Fetch Data
        closes = get_market_data(symbols, start_date, end_date)
        plot_cumulative_returns(closes, saver)

        # Cointegration Analysis
        pvalue_matrix, coint_pairs = find_cointegrated_pairs(closes)
        saver.save_text(f"Cointegrated Pairs: {coint_pairs}", "cointegrated_pairs.txt")
        plot_heatmap(pvalue_matrix, closes.columns, saver)

        # Intraday Data Analysis
        interval = "1m"
        period = "5d"
        start_date = datetime.date(2024, 8, 29)

        # Fetching and Merging Intraday Data
        intraday_data = [
            get_intraday_data(
                symbol, interval=interval, period=period, start_date=start_date
            )
            for symbol in symbols
        ]
        intraday_data = reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True
            ),
            intraday_data,
        )

        # Split Data into Periods
        periods = split_periods(intraday_data)

        # Plot Rolling Correlation for a Specific Pair in a Specific Period
        plot_rolling_correlation(periods["us_mid"], ("ZN", "GC"), saver=saver)
        plot_rolling_correlation(periods["us_mid"], ("ZN", "DXY"), saver=saver)

        # Example: Plot Spread for a Cointegrated Pair
        if coint_pairs:
            symbol1, symbol2 = coint_pairs[0]
            beta = 1.0  # or calculate based on historical regression
            alpha = 0.0  # or calculate based on historical regression
            plot_spread(closes[symbol1], closes[symbol2], beta, alpha, saver)

        # Save dataframes
        saver.save_dataframe(closes, "closes.csv")
        saver.save_dataframe(intraday_data, "intraday_data.csv")
        for period_name, period_data in periods.items():
            saver.save_dataframe(period_data, f"{period_name}_data.csv")
