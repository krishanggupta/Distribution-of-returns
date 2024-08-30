### Step 1: Import Required Libraries
import datetime
from datetime import date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import os


### Step 2: Fetching and Preprocessing Data
def get_market_data(symbols, start_date, end_date, interval="1d", return_change=False):
    closing_prices = yf.download(
        symbols, start=start_date, end=end_date, interval=interval
    )["Adj Close"]
    closing_prices.columns = (
        closing_prices.columns.str.replace("=F", "")
        .str.replace("DX-Y.NYB", "DXY")
        .str.replace("^", "")
    )
    if return_change:
        return closing_prices.pct_change().dropna()
    return closing_prices.dropna()


def get_intraday_data(symbol, interval="5m", period="60d", start_date=None):
    if start_date:
        end_date = start_date - datetime.timedelta(days=int(period.replace("d", "")))
        df = yf.download(symbol, start=end_date, end=start_date, interval=interval)
    else:
        df = yf.Ticker(symbol).history(period=period, interval=interval)

    # Modify column names to remove '=F' or other extensions
    simplified_name = (
        symbol.replace("=F", "").replace("DX-Y.NYB", "DXY").str.replace("^", "")
    )
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


def get_company_name(ticker):
    try:
        company = yf.Ticker(ticker)
        return company.info["longName"]
    except:
        try:
            company = yf.Ticker(ticker)
            return company.info["shortName"]
        except:
            return ticker  # Return the original ticker if unable to fetch the name


start_date = "2021-01-01"
end_date = date.today().strftime("%Y-%m-%d")
symbols = ["ZN=F", "DX-Y.NYB", "CL=F", "GC=F"]
dict_tickers = {
    "ZN=F": "10-Year T-Note Futures",
    "DX-Y.NYB": "US Dollar Index",
    "CL=F": "Crude Oil futures",
    "GC=F": "Gold futures",
    "NQ=F": "Nasdaq 100 futures",
    "^DJI": "Dow Jones Industrial Average",
    "^GSPC": "S&P 500",
}
dict_symbols = {
    "ZN": "10-Year T-Note Futures",
    "DXY": "US Dollar Index",
    "CL": "Crude Oil futures",
    "GC": "Gold futures",
    "NQ": "Nasdaq 100 futures",
    "DJI": "Dow Jones Industrial Average",
    "GSPC": "S&P 500",
}

# Fetch Data
closes = get_market_data(symbols, start_date, end_date)

###Step3: Creating returns of daily close prices

# Create output directory if it doesn't exist
output_dir = "output_daily_close_returns"
os.makedirs(output_dir, exist_ok=True)

# Plot daily returns
returns = closes.pct_change().dropna()

# Save distribution of daily returns
name = f"{start_date}_to_{end_date}"
output_file = os.path.join(output_dir, "Daily_close_returns_" + f"{name}.csv")
returns.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_csv(output_file)

# Plot the returns
ax = returns.plot(figsize=(12, 6))

# Update legend with full names
handles, labels = ax.get_legend_handles_labels()
new_labels = [
    dict_symbols[label.replace("=F", "").replace("^", "").replace("DX-Y.NYB", "DXY")]
    for label in labels
]
ax.legend(handles, new_labels)
plt.title(
    f"Daily Returns of Various Financial Instruments ({start_date} to {end_date})"
)
plt.xlabel("Date")
plt.ylabel("Daily Return")
output_file = os.path.join(output_dir, "Daily_close_returns_" + f"{name}.jpg")
plt.savefig(output_file)
plt.show()
print(f"Returns distribution saved to {output_file}")

###Step4: Intraday returns (High - Low)

# Get today's date
today = datetime.datetime.now()
start_intraday = today.strftime("%Y-%m-%d")

# Calculate the date one week ago
one_week_ago = today - timedelta(days=7)
end_intraday = one_week_ago.strftime("%Y-%m-%d")

# Print the results
print("Today's date:", start_intraday)
print("Date one week ago:", end_intraday)

# Save distribution of daily returns
# Create output directory if it doesn't exist
output_dir = "output_weekly_intraday"
os.makedirs(output_dir, exist_ok=True)
name = f"{end_intraday}_to_{start_intraday}"

# Fetch Data
symbol = "ZN=F"
try:
    data = yf.download(symbol, start=end_intraday, end=start_intraday, interval="1m")

    if data.empty:
        raise ValueError("No data available for the specified date range.")

    # Print some statistics
    print("\nLast one week - Intraday data Summary:")
    print(data.describe())
    output_file = os.path.join(
        output_dir, "Intraday_weekly_summary_" + symbol + f"{name}.csv"
    )
    data.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_csv(output_file)

    # Plot the data
    plt.figure(figsize=(12, 6))

    # Plot Adj Close as a line
    plt.plot(data.index, data["Close"], label="Close", color="green")

    # Shade the area between High and Low
    plt.fill_between(
        data.index,
        data["Low"],
        data["High"],
        alpha=0.1,
        color="gray",
        label="High-Low Range",
    )

    plt.title(
        f"{symbol} Price Data ({data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})"
    )
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_file = os.path.join(
        output_dir, "Weekly_intraday_returns_" + symbol + f"{name}.jpg"
    )
    plt.savefig(output_file)
    plt.show()

    # Print the actual date range of the data
    print(f"\nActual date range of data:")
    print(f"Start: {data.index[0]}")
    print(f"End: {data.index[-1]}")
    print(f"Total periods: {len(data)}")

except Exception as e:
    print(f"An error occurred: {e}")
    print(
        "Try adjusting the date range or check the availability of data for this symbol."
    )
