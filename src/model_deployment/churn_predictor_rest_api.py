"""
The module implement a REST API service with FastAPI for churn prediction service.
"""
# Import Standard Libraries
import os
import mlflow
import pandas as pd
from pathlib import Path
from fastapi import FastAPI

# Import Package Modules
from src.general_utils.general_utils import read_configuration

# Define resources
model_deployment_config, loaded_model, database = None, None, None

# Instance FastAPI object
app = FastAPI()


@app.on_event("startup")
def load_resources():
    """

    :return:
    """
    global model_deployment_config
    global loaded_model
    global database

    # Read configuration
    model_deployment_config = read_configuration(Path(os.getcwd()).parents[0] /
                                                 'configuration' /
                                                 'config.yaml')['model_deployment_config']

    # Load the model
    loaded_model = mlflow.sklearn.load_model(Path(os.getcwd()).parents[0].as_posix()
                                             + '/'
                                             + model_deployment_config['model_path'])

    # Read database
    database = pd.read_csv(Path(os.getcwd()).parents[0].as_posix()
                           + '/'
                           + model_deployment_config['database_path'])


@app.post("/predict")
def predict(customer_id: str) -> dict:
    """

    :param customer_id:
    :return:
    """

    customer = database[database['customerID'] == customer_id]

    prediction = loaded_model.predict(customer.drop(columns=['customerID']))

    # Switch churn prediction
    match prediction[0]:
        case 0:

            return {"prediction": "No Churn, alles gut!"}

        case 1:
            return {"prediction": "Churn, schade!"}
