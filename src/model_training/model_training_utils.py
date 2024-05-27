"""
The module contains utility function for the Model Training pipeline
"""
# Import Standard Libraries
import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# Import Package Modules
from src.logging_module.logging_module import get_logger

# Setup logger
logger = get_logger(os.path.basename(__file__).split('.')[0],
                    pathlib.Path(__file__).parents[1] /
                    'logging_module' /
                    'log_configuration.yaml')


def compute_classification_metrics(y_predicted: np.ndarray,
                                   probabilities: np.ndarray,
                                   y_true: pd.DataFrame,
                                   metrics: list,
                                   round_precision: int = 2) -> pd.DataFrame:
    """
    Compute classification metrics

    Args:
        y_predicted: Numpy array containing the predicted values
        probabilities: Numpy array containing the predicted probabilities
        y_true: Pandas dataframe containing the true values
        metrics: List of metrics to use to compute the classification metrics
        round_precision: integer used to round precision (Default value = 2)

    Returns:
        computed_metrics: Pandas dataframe containing the computed metrics
    """

    logger.info('compute_classification_metrics - Start')

    # Initialise return DataFrame
    computed_metrics = pd.DataFrame(columns=['Value'])

    # Fetch the metrics to evaluate
    if 'Accuracy' in metrics:
        # Compute Accuracy
        accuracy = round(accuracy_score(y_true, y_predicted), round_precision)
        computed_metrics.loc['Accuracy'] = accuracy

    if 'Precision' in metrics:
        # Compute Precision
        precision = round(precision_score(y_true, y_predicted), round_precision)
        computed_metrics.loc['Precision'] = precision

    if 'Recall' in metrics:
        # Compute Recall
        recall = round(recall_score(y_true, y_predicted), round_precision)
        computed_metrics.loc['Recall'] = recall

    if 'F1 Score' in metrics:
        # Compute F1 Score
        f1_score_value = round(f1_score(y_true, y_predicted), round_precision)
        computed_metrics.loc['F1 Score'] = f1_score_value

    if 'ROC AUC' in metrics:
        # Compute ROC AUC
        roc_auc_value = round(roc_auc_score(y_true, probabilities), round_precision)
        computed_metrics.loc['ROC AUC'] = roc_auc_value

    logger.info('compute_classification_metrics - Compute metrics %s', computed_metrics)

    logger.info('compute_classification_metrics - End')

    return computed_metrics
