import pandas as pd
from prophet import Prophet

from models.base_model import Model
from models.prophet.configs import ProphetConfig


class ProphetModel(Model):
    """Prophet model for time series forecasting"""

    def __init__(self, model_name="prophet", config=ProphetConfig(), debug=False):
        super().__init__(model_name=model_name, debug=debug)
        self.config = config  # Use the configuration class
        self.model = Prophet(
            growth=self.config.growth,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            yearly_seasonality=self.config.yearly_seasonality,  # type: ignore
            weekly_seasonality=self.config.weekly_seasonality,  # type: ignore
            daily_seasonality=self.config.daily_seasonality,  # type: ignore
            seasonality_mode=self.config.seasonality_mode,
        )

    def train(self, data: pd.DataFrame):
        df = data[["date", "close"]].copy()

        if self.config.remove_timezone:
            df["date"] = df["date"].dt.tz_localize(None)  # Remove timezone if present

        df = df.rename(columns={"date": "ds", "close": "y"})
        self.model.fit(df)
        self.save()

    def inference(self, input_data: pd.DataFrame) -> pd.DataFrame:
        if self.debug:
            print("Input Data for ProphetModel before predictions:")
            print(input_data)

        # Ensure the 'date' column exists in the input data before proceeding
        if "date" not in input_data.columns:
            raise KeyError(
                "The input_data must contain a 'date' column for Prophet predictions."
            )

        # Rename 'date' to 'ds' for Prophet
        future = input_data[["date"]].rename(columns={"date": "ds"}).copy()

        if self.config.remove_timezone:
            future["ds"] = pd.to_datetime(future["ds"]).dt.tz_localize(None)

        if self.debug:
            print("Future DataFrame for ProphetModel (after renaming):")
            print(future)

        # Perform the forecast using Prophet
        forecast = self.model.predict(future)
        if self.debug:
            print("Forecast Output:")
            print(forecast[["ds", "yhat"]])

        return pd.DataFrame({"date": forecast["ds"], "prediction": forecast["yhat"]})

    def forecast(self, steps: int) -> pd.DataFrame:
        future_dates = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future_dates)
        return forecast[["ds", "yhat"]].tail(steps)
