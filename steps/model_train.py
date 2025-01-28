import logging
from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin


from src.model_dev import LinearRegressionModel
import mlflow
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict,
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        # model = None
        # tuner = None

        # if config.model_name == "lightgbm":
        #     mlflow.lightgbm.autolog()
        #     model = LightGBMModel()
        # elif config.model_name == "randomforest":
        #     mlflow.sklearn.autolog()
        #     model = RandomForestModel()
        # elif config.model_name == "xgboost":
        #     mlflow.xgboost.autolog()
        #     model = XGBoostModel()
        if config["model_name"] == "LinearRegrassion":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
       

        # tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        # if config.fine_tuning:
        #     best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train)
        # else:
        #     trained_model = model.train(x_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config["model_name"]))
    except Exception as e:
        logging.error(e)
        raise e
