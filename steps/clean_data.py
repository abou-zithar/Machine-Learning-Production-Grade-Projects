import logging
from zenml import step
import pandas as pd
from src.data_cleaning import DataCleaning,DataDevideStrategy,DataPreProcessStrategy
from typing_extensions import Annotated

from typing import Tuple
@step
def clean_df(df: pd.DataFrame) ->Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
        Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        process_strategy =DataPreProcessStrategy()
        data_cleaning= DataCleaning(df,process_strategy)
        preprocessed_data=data_cleaning.handle_data()

        divide_strategy = DataDevideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("data cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("error in cleaning data {}".format(e))
        raise e
