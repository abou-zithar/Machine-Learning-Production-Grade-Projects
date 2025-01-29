from pipelines.trainning_pipeline import training_pipeline
from zenml.client import Client


if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="D:\Projects Computer Vision\MLops Projects\My Work\data\olist_customers_dataset.csv")
# mlflow ui --backend-store-uri file:C:\Users\Mahmoud\AppData\Roaming\zenml\local_stores\ec157fe4-1645-4ba8-b3a8-f438087c3fc4\mlruns