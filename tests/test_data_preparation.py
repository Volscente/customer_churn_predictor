"""
This test module includes all the tests for the
module src.data_preparation
"""
# Import Standard Modules
from typing import Union
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Import Package Modules
from src.data_preparation.data_preparation_utils import (
    build_numerical_data_pipeline_steps,
    build_categorical_data_pipeline_steps
)


@pytest.mark.parametrize('step, expected_step, expected_module', [
    (0, 'imputation', SimpleImputer)
])
def test_build_numerical_data_pipeline_steps(fixture_numerical_data_transformations: dict,
                                             step: int,
                                             expected_step: str,
                                             expected_module: SimpleImputer) -> bool:
    """
    Test src.data_preparation.data_preparation_utils.build_numerical_data_pipeline_steps
    by checking the correct instance of data pipelines step objects

    Args:
        fixture_numerical_data_transformations: Dictionary of numerical data transformations configuration
        step: Integer step number
        expected_step: String expected step name
        expected_module: SimpleImputer expected module type

    Returns:
    """

    # Create the data pipelines steps
    numerical_data_pipeline_steps = build_numerical_data_pipeline_steps(fixture_numerical_data_transformations)

    # Retrieve step name and module type
    step_name = numerical_data_pipeline_steps[step][0]
    module_type = type(numerical_data_pipeline_steps[step][1])

    assert step_name == expected_step and module_type == expected_module


@pytest.mark.parametrize('step, expected_step, expected_module', [
    (0, 'imputation', SimpleImputer),
    (1, 'one_hot_encoding', OneHotEncoder)
])
def test_build_categorical_data_pipeline_steps(fixture_categorical_data_transformations: dict,
                                               step: int,
                                               expected_step: str,
                                               expected_module: Union[SimpleImputer, OneHotEncoder]) -> bool:
    """
    Test src.data_preparation.data_preparation_utils.build_categorical_data_pipeline_steps
    by checking the correct instance of data pipelines step objects

    Args:
        fixture_categorical_data_transformations: Dictionary of categorical data transformations configuration
        step: Integer step number
        expected_step: String expected step name
        expected_module: SimpleImputer|OneHotEncoder expected module type

    Returns:
    """

    # Create the data pipelines steps
    categorical_data_pipeline_steps = build_categorical_data_pipeline_steps(fixture_categorical_data_transformations)

    # Retrieve step name and module type
    step_name = categorical_data_pipeline_steps[step][0]
    module_type = type(categorical_data_pipeline_steps[step][1])

    assert step_name == expected_step and module_type == expected_module
