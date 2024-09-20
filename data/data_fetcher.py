# pylint: disable=R0801
import pandas as pd
import requests

from data.config import API_KEYS, BASE_URLS


class DataFetcher:
    """
    A class to fetch and normalize data for Bitcoin, Ethereum, and USD exchange rates.
    """

    def fetch_bitcoin_data(self):
        """Fetch Bitcoin historical data from CoinGecko."""
        url = f"{BASE_URLS['coingecko']}/coins/bitcoin/market_chart?vs_currency=usd&days=max"
        response = requests.get(url, timeout=10)  # Added timeout
        data = response.json()
        return self._normalize_crypto_data(data, "bitcoin")

    def fetch_eth_data(self):
        """Fetch Ethereum historical data from CoinGecko."""
        url = f"{BASE_URLS['coingecko']}/coins/ethereum/market_chart?vs_currency=usd&days=max"
        response = requests.get(url, timeout=10)  # Added timeout
        data = response.json()
        return self._normalize_crypto_data(data, "ethereum")

    def fetch_usd_data(self):
        """Fetch USD historical exchange rate data from Alpha Vantage."""
        url = f"{BASE_URLS['alpha_vantage']}?function=FX_DAILY&from_symbol=USD&to_symbol=EUR&apikey={API_KEYS['alpha_vantage']}"
        response = requests.get(url, timeout=10)  # Added timeout
        data = response.json()
        return self._normalize_currency_data(data)

    def _normalize_crypto_data(self, data, asset_name):
        """Normalize crypto data to match the required schema."""
        prices = data.get("prices", [])
        if not prices:
            print(f"No price data available for {asset_name}")
            return pd.DataFrame()  # Return an empty DataFrame if no data is found

        try:
            normalized_data = pd.DataFrame(prices, columns=["timestamp", "close"])
        except ValueError as e:
            print(f"Error in processing data for {asset_name}: {e}")
            return pd.DataFrame()

        normalized_data["date"] = pd.to_datetime(
            normalized_data["timestamp"], unit="ms"
        )
        normalized_data["open"] = normalized_data["close"].shift(1)
        normalized_data["high"] = normalized_data[
            "close"
        ]  # Simplified for this example
        normalized_data["low"] = normalized_data["close"]  # Simplified for this example
        normalized_data["volume"] = (
            0  # Volume data isn't available in this API response
        )
        normalized_data["asset"] = asset_name
        normalized_data = normalized_data.drop(columns=["timestamp"])

        normalized_data = pd.DataFrame(
            normalized_data,
            columns=["date", "open", "high", "low", "close", "volume", "asset"],
        )
        return normalized_data

    def _normalize_currency_data(self, data):
        """Normalize USD exchange rate data."""
        time_series = data.get("Time Series FX (Daily)", {})
        if not time_series:
            print("No time series data available for USD")
            return pd.DataFrame()  # Return an empty DataFrame if no data is found

        normalized_data = pd.DataFrame(time_series).T
        normalized_data["date"] = pd.to_datetime(normalized_data.index)
        normalized_data = normalized_data.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
            }
        )
        normalized_data["volume"] = 0  # Forex data doesn't have volume
        normalized_data["asset"] = "USD"

        return normalized_data[
            ["date", "open", "high", "low", "close", "volume", "asset"]
        ]
