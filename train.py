# conda create --name modelmaker python=3.9
# conda activate modelmaker
# pip install cython numpy==1.24.3
# pip install -r requirements.txt

# pylint: disable=no-name-in-module
from configs import models
from data.csv_loader import CSVLoader
from data.data_fetcher import DataFetcher
from data.utils.data_preprocessing import preprocess_data
from models.model_factory import ModelFactory


def select_data(fetcher, default_selection=None, file_path=None):
    """Provide an interface to choose between CSV data or API data."""
    if default_selection is None:
        print("Select the data source:")
        print("1. Bitcoin (BTC) from API")  # Example of a default selection
        print("2. Ethereum (ETH) from API")  # Example of a default selection
        print("3. USD Exchange Rates from API")  # Example of a default selection
        print("4. Load data from CSV file")

        selection = input("Enter your choice (1/2/3/4): ").strip()
    else:
        selection = default_selection

    if selection == "1":
        print("Fetching Bitcoin data from API...")
        return fetcher.fetch_bitcoin_data()
    if selection == "2":
        print("Fetching Ethereum data from API...")
        return fetcher.fetch_eth_data()
    if selection == "3":
        print("Fetching USD Exchange Rate data from API...")
        return fetcher.fetch_usd_data()
    if selection == "4":
        if file_path is None:
            file_path = input("Enter the CSV file path: ").strip()
        return CSVLoader.load_csv(file_path)

    print("Invalid choice, defaulting to Bitcoin from API.")
    return fetcher.fetch_bitcoin_data()


def model_selection_input():
    print("Select the models to train:")
    print("1. All models")
    print("2. Custom selection")

    model_selection = input("Enter your choice (1/2): ").strip()

    if model_selection == "1":
        model_types = models
    elif model_selection == "2":
        available_models = {str(i + 1): model for i, model in enumerate(models)}
        print("Available models to train:")
        for key, value in available_models.items():
            print(f"{key}. {value}")

        selected_models = input(
            "Enter the numbers of the models to train (e.g., 1,3,5): "
        ).strip()
        model_types = [
            available_models[num.strip()]
            for num in selected_models.split(",")
            if num.strip() in available_models
        ]
    else:
        print("Invalid choice, defaulting to all models.")
        model_types = models

    return model_types


def main():
    fetcher = DataFetcher()

    # Select data dynamically based on user input
    data = select_data(fetcher) # example testing defaults , "4", "data/sets/eth.csv"

    # Normalize and preprocess the data
    data = preprocess_data(data)

    # Initialize ModelFactory
    factory = ModelFactory()

    # Select models to train
    model_types = model_selection_input()

    # Train and save the selected models
    for model_type in model_types:
        print(f"Training {model_type} model...")
        model = factory.create_model(model_type)
        model.train(data)

    print("Model training and saving complete!")


if __name__ == "__main__":
    main()
