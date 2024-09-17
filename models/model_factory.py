import importlib

# pylint: disable=import-error, no-name-in-module
from utils.common import snake_to_camel

# pylint: disable=import-error, no-name-in-module
from utils.model_commons import set_seed


class ModelFactory:
    """Factory class to dynamically create and manage machine learning models."""

    def __init__(self, seed=42):
        """Initialize the factory with a default random seed for reproducibility."""
        set_seed(seed)
        self.models_dir = "models"

    def create_model(self, model_name: str):
        """Dynamically import and create a model class based on the model_name."""
        print(f"Initializing model: {model_name}")
        try:
            # Dynamically construct the module path based on model_name
            module_name = f"{self.models_dir}.{model_name}.model"

            # Import the model module dynamically (from model.py)
            model_module = importlib.import_module(module_name)

            # Convert the model_name from snake_case to CamelCase
            model_class_name = snake_to_camel(model_name) + "Model"
            print(f"Model class name: {model_class_name}")

            # Get the model class from the imported module
            model_class = getattr(model_module, model_class_name)

            # Return an instance of the model class
            return model_class()
        except ModuleNotFoundError as e:
            raise ValueError(
                f"Model '{model_name}' not found in {self.models_dir}. Error: {str(e)}"
            ) from e
        except AttributeError as e:
            raise ValueError(
                f"Model class '{model_class_name}' not found in {model_name}.model. Error: {str(e)}"
            ) from e
