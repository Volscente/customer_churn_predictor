"""
This test module includes all the fixtures necessary
for running PyTest tests
"""
# Import Standard Libraries
import pathlib
import pytest

# Import Package Modules
from src.general_utils.general_utils import read_configuration

# Read configuration file
configuration = read_configuration(pathlib.Path(__file__).parents[1]
                                   / 'configuration'
                                   / 'test_config.yaml')


@pytest.fixture
def fixture_numerical_data_transformations(
        test_numerical_data_transformations: dict = configuration['test_numerical_data_transformations']
) -> dict:
    """
    Fixture for a Dictionary Numerical Data Transformations with structure:
        <transformation_name>:
            include: <bool>
            module: <string module name>

    Args:
        test_numerical_data_transformations: dict of numerical data transformations

    Returns:
        test_numerical_data_transformations: dict of numerical data transformations
    """

    return test_numerical_data_transformations


@pytest.fixture
def fixture_categorical_data_transformations(
        test_categorical_data_transformations: dict = configuration['test_categorical_data_transformations']
) -> dict:
    """
    Fixture for a Dictionary Categorical Data Transformations with structure:
        <transformation_name>:
            include: <bool>
            module: <string module name>

    Args:
        test_categorical_data_transformations: dict of categorical data transformations

    Returns:
        test_categorical_data_transformations: dict of categorical data transformations
    """

    return test_categorical_data_transformations


@pytest.fixture
def fixture_classification_metrics(
        test_classification_metrics: list = configuration['test_classification_metrics']
) -> list:
    """
    Fixture for classification metrics list

    Args:
        test_classification_metrics: list of classification metrics

    Returns:
        test_classification_metrics: list of classification metrics
    """

    return test_classification_metrics
