"""
This test module includes all the tests for the
module src.model_training
"""
# Import Standard Modules
import numpy as np
import pytest

# Import Package Modules
from src.model_training.model_training_utils import (
    compute_classification_metrics
)


@pytest.mark.parametrize('y_predicted, probabilities, y_true, expected_metrics', [
    ([1, 1, 0, 0, 1],
     np.array([[0.2, 0.8], [0.2, 0.8], [0.8, 0.2], [0.8, 0.2], [0.2, 0.8]]),
     [1, 1, 0, 0, 0],
     [0.8, 0.67, 1.00, 0.80, 0.83])
])
def test_compute_classification_metrics(y_predicted: np.ndarray,
                                        probabilities: np.array,
                                        y_true: np.ndarray,
                                        expected_metrics: np.ndarray,
                                        fixture_classification_metrics: list) -> bool:
    """
    This function tests the function src.model_training.model_training_utils.compute_classification_metrics
    whether the classification metrics are calculated correctly.

    Args:
        y_predicted: np.ndarray of predicted values
        probabilities: np.array of probabilities
        y_true: np.ndarray of true values
        expected_metrics: np.ndarray of expected classification metrics values
        fixture_classification_metrics: list of metrics

    Returns:
    """

    # Apply the function to test and compute the metrics
    metrics = compute_classification_metrics(y_predicted,
                                             probabilities,
                                             y_true,
                                             fixture_classification_metrics)

    # Reshaping metrics for comparison
    metrics = metrics.values.reshape(1, -1)[0]

    assert np.array_equal(metrics, expected_metrics)
