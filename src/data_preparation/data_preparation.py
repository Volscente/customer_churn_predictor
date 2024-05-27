"""
The module contains object classes for the Data Preparation pipelines and components
"""
# Import Standard Libraries
import pathlib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import Package Modules
from src.logging_module.logging_module import get_logger
from src.data_preparation.data_preparation_utils import (
    build_numerical_data_pipeline_steps,
    build_categorical_data_pipeline_steps
)


class CustomerChurnDataPreparation:
    """
    The class implements a Training Data Preparation pipeline,
    which involves standard data preparation techniques
    like imputation, normalisation and standardisation.

    Attributes:
        logger: logging.Logger object for log messages
        data_transformations: Dictionary of data preparation transformations to apply
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        numerical_data_pipeline_steps: List of numerical data pipeline steps
        categorical_data_pipeline_steps: List of categorical data pipeline steps
    """

    def __init__(self,
                 data_transformations: dict,
                 features: dict):
        """
        The constructor of the TrainingDataPreparation object
        initialise the data preparation transformation
        dictionary and numerical and categorical features list.

        Args:
            data_transformations: Dictionary of data preparation transformations to apply
            features: Dictionary of features list 'numerical' and 'categorical'
        """
        # Setup logger
        self.logger = get_logger(__class__.__name__,
                                 pathlib.Path(__file__).parents[1] /
                                 'logging_module' /
                                 'log_configuration.yaml')

        self.logger.info('__init__ - Initialise object attributes')

        # Initialise attributes
        self.data_transformations = data_transformations

        # Check if there are numerical and/or categorical features
        if 'numerical' in features:
            self.numerical_features = features['numerical']
        if 'categorical' in features:
            self.categorical_features = features['categorical']

        # Initialise data preparation pipeline steps
        self.numerical_data_pipeline_steps = None
        self.categorical_data_pipeline_steps = None

    def build_training_data_preparation_pipeline(self) -> ColumnTransformer:
        """
        Builds the training data preparation pipeline

        Returns:
            training_data_preparation_pipeline: sklearn.compose.ColumnTransformer
                                                with required data preparation steps
        """
        self.logger.info('build_training_data_preparation_pipeline - Start')

        self.logger.info('build_training_data_preparation_pipeline - Build the Numerical Data Pipeline')

        # Define the numerical data pipeline steps
        self.numerical_data_pipeline_steps = build_numerical_data_pipeline_steps(
            self.data_transformations['numerical']
        )

        self.logger.info('build_training_data_preparation_pipeline - Build the Categorical Data Pipeline')

        # Define the categorical data pipeline steps
        self.categorical_data_pipeline_steps = build_categorical_data_pipeline_steps(
            self.data_transformations['categorical']
        )

        self.logger.info('build_training_data_preparation_pipeline - Bundle the data pipeline')

        # Bundle the data pipeline
        training_data_preparation_pipeline = ColumnTransformer(
            transformers=[
                ('numerical', Pipeline(self.numerical_data_pipeline_steps), self.numerical_features),
                ('categorical', Pipeline(self.categorical_data_pipeline_steps), self.categorical_features)
            ]
        )

        self.logger.info('build_training_data_preparation_pipeline - End')

        return training_data_preparation_pipeline
