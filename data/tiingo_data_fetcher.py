import os

import pandas as pd
import requests
from dotenv import load_dotenv

# Load the .env.local file if it exists, otherwise load .env
if os.path.exists(".env.local"):
    print("Loading .env.local file...")
    load_dotenv(dotenv_path=".env.local", override=True)
else:
    print("Loading .env file...")
    load_dotenv(dotenv_path=".env")  # Defaults to loading .env

# Retrieve the API keys from environment variables
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
BASE_APIURL = "https://api.tiingo.com"


class DataFetcher:
    """
    A class to fetch and normalize data for stocks and cryptocurrencies from Tiingo.
    """

    def __init__(self, cache_folder="data/sets"):
        self.cache_folder = cache_folder
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)  # Ensure the 'sets' folder exists

    def _generate_filename(self, symbol, start_date, end_date, frequency):
        """Generate a unique filename for the CSV based on the symbol and parameters."""
        return os.path.join(
            self.cache_folder, f"{symbol}_{start_date}_to_{end_date}_{frequency}.csv"
        )

    def fetch_tiingo_stock_data(self, symbol, start_date, end_date, frequency="daily"):
        """Fetch historical stock data from Tiingo."""

        filename = self._generate_filename(symbol, start_date, end_date, frequency)

        # Check if the CSV file already exists
        if os.path.exists(filename):
            print(f"Loading stock data from {filename}...")
            return pd.read_csv(filename)

        # Define the URL, headers, and parameters for the request
        url = f"{BASE_APIURL}/tiingo/daily/{symbol}/prices"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {TIINGO_API_KEY}",
        }
        params = {
            "startDate": start_date,
            "endDate": end_date,
            # daily: Values returned as daily periods, with a holiday calendar.
            # weekly: Values returned as weekly data, with days ending on Friday.
            # monthly: Values returned as monthly data, with days ending on the last standard business day (Mon-Fri) of each month.
            # annually: Values returned as annual data, with days ending on the last standard business day (Mon-Fri) of each year.
            "resampleFreq": frequency,
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code != 200:
            print(
                f"Error fetching stock data from Tiingo for {symbol}: {response.status_code}"
            )
            return pd.DataFrame()  # Return an empty DataFrame if there’s an error

        data = response.json()
        df = self._normalize_tiingo_data(data, symbol)

        # Save the fetched data to CSV
        print(f"Saving stock data to {filename}...")
        df.to_csv(filename, index=False)

        return df

    def fetch_tiingo_crypto_data(self, symbol, start_date, end_date, frequency="5min"):
        """Fetch historical cryptocurrency data from Tiingo."""

        filename = self._generate_filename(symbol, start_date, end_date, frequency)

        # Check if the CSV file already exists
        if os.path.exists(filename):
            print(f"Loading stock data from {filename}...")
            return pd.read_csv(filename)

        # Define the URL, headers, and parameters for the request
        url = f"{BASE_APIURL}/tiingo/crypto/prices"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {TIINGO_API_KEY}",
        }
        params = {
            "tickers": symbol,
            "startDate": start_date,
            "endDate": end_date,
            # The minimum value is "1min". Units in minutes (min), hours (hour), and days (day) are accepted.
            # Format is # + (min/hour/day); e.g. "15min", "4hour" or "1day".
            # If no value is provided, defaults to 5min.
            "resampleFreq": frequency,
        }

        # Send request to Tiingo API
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return pd.DataFrame()

        # Parse JSON response
        try:
            data = response.json()
            if not data or "priceData" not in data[0]:
                print(f"No crypto data found for {symbol}")
                return pd.DataFrame()
        except (ValueError, KeyError) as e:
            print(f"Error parsing response data: {e}")
            return pd.DataFrame()

        df = self._normalize_tiingo_data(data[0]["priceData"], symbol)

        # Save the fetched data to CSV
        print(f"Saving crypto data to {filename}...")
        df.to_csv(filename, index=False)

        return df

    def _normalize_tiingo_data(self, data, asset_name):
        """Normalize Tiingo stock data to match the required schema."""
        if not data:
            print(f"No data available for {asset_name}")
            return pd.DataFrame()

        try:
            normalized_data = pd.DataFrame(data)
        except ValueError as e:
            print(f"Error in processing data for {asset_name}: {e}")
            return pd.DataFrame()

        normalized_data["date"] = pd.to_datetime(
            normalized_data["date"], errors="coerce"
        )

        return normalized_data[["date", "open", "high", "low", "close", "volume"]]
