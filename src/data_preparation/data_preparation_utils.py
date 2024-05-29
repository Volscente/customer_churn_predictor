"""
The module contains utility function for the Data Preparation pipeline
"""
# Import Standard Libraries
import os
import pathlib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Import Package Modules
from src.logging_module.logging_module import get_logger


# Setup logger
logger = get_logger(os.path.basename(__file__).split('.')[0],
                    pathlib.Path(__file__).parents[1] /
                    'logging_module' /
                    'log_configuration.yaml')


def build_numerical_data_pipeline_steps(numerical_data_transformations: dict) -> list:
    """
    Build numerical data transformations steps based on
    the configuration in 'numerical_data_transformations'

    Args:
        numerical_data_transformations: Dictionary of numerical data transformations configuration

    Returns:
        numerical_data_pipeline_steps: List of numerical data transformation steps
    """

    logger.info('build_numerical_data_pipeline_steps - Start')

    # Initialise numerical data pipeline steps list
    numerical_data_pipeline_steps = []

    logger.info('build_numerical_data_pipeline_steps - Building steps')

    # 1. Check feature engineering step
    if numerical_data_transformations['feature_engineering']['include']:
        pass
    else:
        logger.info('build_numerical_data_pipeline_steps - Skipping Feature Engineering step')

    # 2. Check imputation step
    if numerical_data_transformations['imputation']['include']:

        # Retrieve imputation module to use
        imputation_module = numerical_data_transformations['imputation']['module']

        logger.info('build_numerical_data_pipeline_steps - Adding %s Imputation step',
                    imputation_module)

        # Switch the imputation technique to apply
        match imputation_module:
            case 'SimpleImputer':
                numerical_data_pipeline_steps.append(
                    ('imputation',
                     SimpleImputer(strategy='median', copy=False))
                )
    else:
        logger.info('build_numerical_data_pipeline_steps - Skipping Imputation step')

    # 3. Check standardisation step
    if numerical_data_transformations['standardisation']['include']:

        # Retrieve standardisation module to use
        standardisation_module = numerical_data_transformations['standardisation']['module']

        logger.info('build_numerical_data_pipeline_steps - Adding %s Standardisation step',
                    standardisation_module)

        # Switch the imputation technique to apply
        match standardisation_module:
            case 'MinMaxScaler':
                numerical_data_pipeline_steps.append(
                    ('standardisation',
                     MinMaxScaler())
                )
    else:
        logger.info('build_numerical_data_pipeline_steps - Skipping Standardisation step')

    # 4. Check normalization step
    if numerical_data_transformations['normalization']['include']:
        pass
    else:
        logger.info('build_numerical_data_pipeline_steps - Skipping Normalization step')

    logger.info('build_numerical_data_pipeline_steps - End')

    return numerical_data_pipeline_steps


def build_categorical_data_pipeline_steps(categorical_data_transformations: dict) -> list:
    """
    Build categorical data transformations steps based
    on the configuration in 'categorical_data_transformations'

    Args:
        categorical_data_transformations: Dictionary of categorical
                                          data transformations configuration

    Returns:
        categorical_data_pipeline_steps: List of categorical data transformation steps
    """
    logger.info('build_categorical_data_pipeline_steps - Start')

    # Initialise categorical data pipeline steps list
    categorical_data_pipeline_steps = []

    logger.info('build_categorical_data_pipeline_steps - Building steps')

    # 1. Check imputation step
    if categorical_data_transformations['imputation']['include']:

        # Retrieve imputation module to use
        imputation_module = categorical_data_transformations['imputation']['module']

        logger.info('build_categorical_data_pipeline_steps - Adding %s Imputation step',
                    imputation_module)

        # Switch the imputation technique to apply
        match imputation_module:
            case 'SimpleImputer':
                categorical_data_pipeline_steps.append(
                    ('imputation',
                     SimpleImputer(strategy='constant', fill_value='unknown', copy=False))
                )
    else:
        logger.info('build_categorical_data_pipeline_steps - Skipping Imputation step')

    # 2. Check one hot encoding step
    if categorical_data_transformations['one_hot_encoding']['include']:

        # Retrieve one hot encoding module to use
        one_hot_encoding_module = categorical_data_transformations['one_hot_encoding']['module']

        logger.info('build_categorical_data_pipeline_steps - Adding %s One-hot Encoding step',
                    one_hot_encoding_module)

        # Switch the one hot encoding technique to apply
        match one_hot_encoding_module:
            case 'OneHotEncoder':
                categorical_data_pipeline_steps.append(
                    ('one_hot_encoding',
                     OneHotEncoder())
                )
    else:
        logger.info('build_categorical_data_pipeline_steps - Skipping One-hot Encoding step')

    logger.info('build_categorical_data_pipeline_steps - End')

    return categorical_data_pipeline_steps
