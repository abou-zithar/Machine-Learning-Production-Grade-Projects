from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from  steps.evaluation import evaluate_model
from steps.model_train import train_model
from steps import config
@pipeline(enable_cache=True)
def training_pipeline(data_path : str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test =clean_df(df)
    
    model = train_model(X_train, X_test, y_train, y_test,{"model_name":"LinearRegrassion", "fine_tuning":False} )
    r2,rmse = evaluate_model(model,X_test,y_test)

