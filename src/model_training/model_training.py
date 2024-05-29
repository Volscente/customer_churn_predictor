"""
The module contains object classes for the Model Training pipelines and components
"""
# Import Standard Libraries
import pathlib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import Package Modules
from src.logging_module.logging_module import get_logger
from src.model_training.model_training_utils import compute_classification_metrics


class ModelTrainer:
    """
    The class implements a Model Training pipeline,
    which involves standard model training pipeline bundle,
    estimator fit and evaluation.

    Attributes:
        logger: logging.Logger object for log messages
        model_name: String name of the model to be trained
        model: Classification model to be fitted and evaluated
        data_pipeline: ColumnTransformer with required data preparation steps
        pipeline: Scikit-learn Pipeline object that bundles the model and data_pipeline
    """

    def __init__(self,
                 model_name: str,
                 model: LogisticRegression,
                 data_pipeline: ColumnTransformer):
        """
        The constructor of the ModelTrainer object.

        Args:
            model_name: String name of the model to be trained
            model: Classification model to be fitted and evaluated
            data_pipeline: ColumnTransformer with required data preparation steps
        """
        # Setup logger
        self.logger = get_logger(__class__.__name__,
                                 pathlib.Path(__file__).parents[1] /
                                 'logging_module' /
                                 'log_configuration.yaml')

        self.logger.info('__init__ - Initialise object attributes')

        # Initialise attributes
        self.model_name = model_name
        self.model = model
        self.data_pipeline = data_pipeline
        self.pipeline = None

    def bundle_and_fit_pipeline(self,
                                x: pd.DataFrame,
                                y: pd.DataFrame):
        """
        Bundles the data pipeline and model into a Pipeline and perform the training

        Args:
            x: Pandas Dataframe of features values
            y: Pandas Dataframe of labels values

        Returns:
        """
        self.logger.info('bundle_and_fit_pipeline - Start')

        self.logger.info('bundle_and_fit_pipeline - Bundle the pipeline')

        # Bundle the data_pipeline and the model together into a Pipline object
        self.pipeline = Pipeline([
            ('data_preprocessing', self.data_pipeline),
            (self.model_name, self.model)
        ])

        self.logger.info('bundle_and_fit_pipeline - Fit the pipeline')

        # Train the pipeline
        self.pipeline.fit(x, y)

        self.logger.info('bundle_and_fit_pipeline - End')

    def evaluate_pipeline(self,
                          x: pd.DataFrame,
                          y: pd.DataFrame,
                          metrics: list) -> pd.DataFrame:
        """
        Bundles the data pipeline and model into a Pipeline and perform the training

        Args:
            x: Pandas Dataframe of features values
            y: Pandas Dataframe of labels values
            metrics: List of metrics to use in the evaluation

        Returns:
            evaluation: Pandas Dataframe of evaluation metrics results
        """
        self.logger.info('evaluate_pipeline - Start')

        self.logger.info('evaluate_pipeline - Compute predictions')

        try:

            # Compute predictions
            predictions = self.pipeline.predict(x)

            # Compute prediction probabilities
            prediction_probabilities = self.pipeline.predict_proba(x)

        except AttributeError as exc:

            self.logger.error('evaluate_pipeline - Impossible to compute predictions')

            raise AttributeError('evaluate_pipeline - Ensure to run first bundle_and_fit_pipeline') from exc

        self.logger.info('evaluate_pipeline - Evaluate pipeline')

        # Compute evaluation metrics
        evaluation = compute_classification_metrics(y_predicted=predictions,
                                                    probabilities=prediction_probabilities,
                                                    y_true=y,
                                                    metrics=metrics)

        self.logger.info('evaluate_pipeline - End')

        return evaluation
