"""
Evaluator Module for CFST XGBoost Pipeline

This module handles model evaluation and metrics calculation for regression tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Model evaluator for regression tasks.

    Calculates comprehensive metrics including RMSE, MAE, R², and MAPE.
    Generates evaluation reports and handles metric persistence.
    """

    def __init__(self):
        """Initialize Evaluator."""
        self.metrics_history = []
        logger.info("Evaluator initialized")

    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True target values
            y_pred: Predicted target values

        Returns:
            Dictionary containing all calculated metrics

        Raises:
            ValueError: If input arrays have different lengths
        """
        logger.info("Calculating evaluation metrics")

        # Validate inputs
        if len(y_true) != len(y_pred):
            error_msg = f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Convert to numpy arrays for calculations
        y_true_array = np.array(y_true).flatten()
        y_pred_array = np.array(y_pred).flatten()

        # Check for NaN values
        if np.isnan(y_true_array).any() or np.isnan(y_pred_array).any():
            logger.warning("NaN values detected in predictions or targets")

        # Calculate metrics
        try:
            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(mean_squared_error(y_true_array, y_pred_array))

            # MAE (Mean Absolute Error)
            mae = mean_absolute_error(y_true_array, y_pred_array)

            # R² (Coefficient of Determination)
            r2 = r2_score(y_true_array, y_pred_array)

            # MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero
            mask = y_true_array != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true_array[mask] - y_pred_array[mask]) / y_true_array[mask])) * 100
            else:
                mape = np.nan
                logger.warning("Cannot calculate MAPE: all true values are zero")

            # Additional metrics
            mse = mean_squared_error(y_true_array, y_pred_array)

            # Calculate max error
            max_error = np.max(np.abs(y_true_array - y_pred_array))

            # Calculate mean prediction
            mean_pred = np.mean(y_pred_array)
            mean_true = np.mean(y_true_array)

            # Create metrics dictionary
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape) if not np.isnan(mape) else None,
                'mse': float(mse),
                'max_error': float(max_error),
                'mean_prediction': float(mean_pred),
                'mean_actual': float(mean_true),
                'n_samples': len(y_true_array)
            }

            # Log metrics
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R²: {r2:.4f}")
            if not np.isnan(mape):
                logger.info(f"MAPE: {mape:.2f}%")
            logger.info(f"Max Error: {max_error:.4f}")

            return metrics

        except Exception as e:
            error_msg = f"Failed to calculate metrics: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a trained model on given data.

        Args:
            model: Trained model with predict method
            X: Features DataFrame
            y: True target values
            model_name: Name of the model for logging

        Returns:
            Dictionary containing metrics and evaluation info
        """
        logger.info(f"Evaluating {model_name} on {len(X)} samples")

        try:
            # Make predictions
            start_time = pd.Timestamp.now()
            y_pred = model.predict(X)
            prediction_time = (pd.Timestamp.now() - start_time).total_seconds()

            # Calculate metrics
            metrics = self.calculate_metrics(y, y_pred)

            # Add evaluation metadata
            evaluation_result = {
                'model_name': model_name,
                'timestamp': pd.Timestamp.now().isoformat(),
                'metrics': metrics,
                'prediction_time': prediction_time,
                'prediction_time_per_sample': prediction_time / len(X) if len(X) > 0 else 0
            }

            # Store in history
            self.metrics_history.append(evaluation_result)

            logger.info(f"Model evaluation completed in {prediction_time:.4f} seconds")

            return evaluation_result

        except Exception as e:
            error_msg = f"Model evaluation failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def cross_validate_evaluation(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                 cv_splits: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.

        Args:
            model: Model to evaluate (will be cloned for each fold)
            X: Features DataFrame
            y: Target Series
            cv_splits: Number of cross-validation folds
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting {cv_splits}-fold cross-validation evaluation")

        from sklearn.model_selection import KFold
        from sklearn.base import clone

        try:
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            fold_metrics = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                logger.debug(f"Evaluating fold {fold + 1}/{cv_splits}")

                # Split data
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]

                # Clone and train model
                fold_model = clone(model)
                fold_model.fit(X_train_fold, y_train_fold)

                # Evaluate on validation set
                y_pred_fold = fold_model.predict(X_val_fold)
                metrics = self.calculate_metrics(y_val_fold, y_pred_fold)

                fold_metrics.append({
                    'fold': fold + 1,
                    'metrics': metrics,
                    'n_train_samples': len(train_idx),
                    'n_val_samples': len(val_idx)
                })

            # Calculate aggregate statistics
            all_rmse = [f['metrics']['rmse'] for f in fold_metrics]
            all_mae = [f['metrics']['mae'] for f in fold_metrics]
            all_r2 = [f['metrics']['r2'] for f in fold_metrics]

            cv_results = {
                'fold_metrics': fold_metrics,
                'aggregate': {
                    'mean_rmse': float(np.mean(all_rmse)),
                    'std_rmse': float(np.std(all_rmse)),
                    'mean_mae': float(np.mean(all_mae)),
                    'std_mae': float(np.std(all_mae)),
                    'mean_r2': float(np.mean(all_r2)),
                    'std_r2': float(np.std(all_r2)),
                    'n_folds': cv_splits
                }
            }

            # Log aggregate results
            agg = cv_results['aggregate']
            logger.info(f"CV RMSE: {agg['mean_rmse']:.4f} (+/- {agg['std_rmse']:.4f})")
            logger.info(f"CV MAE: {agg['mean_mae']:.4f} (+/- {agg['std_mae']:.4f})")
            logger.info(f"CV R²: {agg['mean_r2']:.4f} (+/- {agg['std_r2']:.4f})")

            return cv_results

        except Exception as e:
            error_msg = f"Cross-validation evaluation failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def save_evaluation_report(self, evaluation_result: Dict[str, Any],
                              output_path: str) -> None:
        """
        Save evaluation report to JSON file.

        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Path to save the report

        Raises:
            Exception: If saving fails
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(evaluation_result, f, indent=2)

            logger.info(f"Evaluation report saved to {output_path}")

        except Exception as e:
            error_msg = f"Failed to save evaluation report: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def load_evaluation_report(self, report_path: str) -> Dict[str, Any]:
        """
        Load evaluation report from JSON file.

        Args:
            report_path: Path to the evaluation report

        Returns:
            Evaluation result dictionary

        Raises:
            Exception: If loading fails
        """
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)

            logger.info(f"Evaluation report loaded from {report_path}")
            return report

        except Exception as e:
            error_msg = f"Failed to load evaluation report: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def compare_models(self, model_results: list) -> pd.DataFrame:
        """
        Compare multiple models' evaluation results.

        Args:
            model_results: List of evaluation result dictionaries

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        for result in model_results:
            model_name = result.get('model_name', 'unknown')
            metrics = result.get('metrics', {})

            comparison_data.append({
                'Model': model_name,
                'RMSE': metrics.get('rmse'),
                'MAE': metrics.get('mae'),
                'R²': metrics.get('r2'),
                'MAPE': metrics.get('mape'),
                'Samples': metrics.get('n_samples')
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')

        logger.info("Model comparison completed")
        logger.info(f"\n{comparison_df.to_string(index=False)}")

        return comparison_df

    def get_best_model(self, model_results: list, metric: str = 'rmse') -> Dict[str, Any]:
        """
        Get the best model based on specified metric.

        Args:
            model_results: List of evaluation result dictionaries
            metric: Metric to use for comparison (rmse, mae, r2)

        Returns:
            Best model's evaluation result
        """
        if not model_results:
            logger.warning("No model results provided")
            return {}

        # Determine if higher or lower is better
        lower_is_better = metric in ['rmse', 'mae', 'mape']

        # Find best model
        if lower_is_better:
            best_result = min(model_results, key=lambda x: x.get('metrics', {}).get(metric, float('inf')))
        else:
            best_result = max(model_results, key=lambda x: x.get('metrics', {}).get(metric, float('-inf')))

        best_model_name = best_result.get('model_name', 'unknown')
        best_value = best_result.get('metrics', {}).get(metric)

        logger.info(f"Best model: {best_model_name} ({metric}: {best_value:.4f})")

        return best_result
