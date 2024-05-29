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
from src.logging_module.logging_module import get_logger

# Define resources
model_deployment_config, loaded_model, database = None, None, None

# Setup logger
logger = get_logger(os.path.basename(__file__).split('.')[0],
                    Path(__file__).parents[1] /
                    'logging_module' /
                    'log_configuration.yaml')

# Instance FastAPI object
app = FastAPI()


@app.on_event("startup")
def load_resources():
    """
    Load required resources
    """
    global model_deployment_config
    global loaded_model
    global database

    logger.info('load_resources - Start')

    # Define the root path (Path(os.getcwd()).parents[0] | '/app'/)
    root_path = Path(os.getcwd())

    # Read configuration
    model_deployment_config = read_configuration(root_path /
                                                 'configuration' /
                                                 'config.yaml')['model_deployment_config']

    # Model path
    model_path = root_path.as_posix() + '/' + model_deployment_config['model_path']

    logger.info('load_resources - Loading model from %s', model_path)

    # Load the model
    loaded_model = mlflow.sklearn.load_model(model_path)

    logger.info('load_resources - Model loaded')

    # Database path
    database_path = root_path.as_posix() + '/' + model_deployment_config['database_path']

    logger.info('load_resources - Loading database from %s', database_path)

    # Read database
    database = pd.read_csv(database_path)

    logger.info('load_resources - Database loaded')

    logger.info('load_resources - End')


@app.post("/predict")
def predict(customer_id: str) -> dict:
    """
    Given a customer_id, predict whether the customer is churn or not.
    Args:
        customer_id: String customer id
    """

    logger.info('predict - Start')

    # Retrieve customer information
    customer = database[database['customerID'] == customer_id]

    logger.info('predict - Computing prediction for customer %s', customer_id)

    # Predict
    prediction = loaded_model.predict(customer.drop(columns=['customerID']))

    # Switch churn prediction
    match prediction[0]:
        case 0:

            return {"prediction": "No Churn, alles gut!"}

        case 1:
            return {"prediction": "Churn, schade!"}
