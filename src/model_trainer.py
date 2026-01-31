'''
Model Trainer Module for CFST XGBoost Pipeline

This module handles XGBoost model training, cross-validation, and hyperparameter optimization.
'''

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import optuna
import time
import json
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.model_utils import load_best_params, save_best_params

logger = get_logger(__name__)

class ModelTrainer:
    '''
    Model trainer for CFST XGBoost pipeline.

    Handles XGBoost model training, cross-validation, and optional hyperparameter optimization.
    '''

    def __init__(self, params: Optional[Dict[str, Any]] = None, use_optuna: bool = False,
                 n_trials: int = 100, optuna_timeout: int = 3600):
        '''
        Initialize ModelTrainer.

        Args:
            params: XGBoost parameters (defaults to reasonable values if None)
            use_optuna: Whether to use Optuna for hyperparameter optimization
            n_trials: Number of Optuna trials for hyperparameter optimization
            optuna_timeout: Timeout for Optuna optimization in seconds
        '''
        self.params = params or self._get_default_params()
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.optuna_timeout = optuna_timeout
        self.model: Optional[xgb.XGBRegressor] = None
        self.training_history = []

        # Auto-load best parameters if use_optuna is False
        if not self.use_optuna:
            loaded_params = load_best_params()
            if loaded_params is not None:
                self.params.update(loaded_params)
                logger.info("Using loaded best parameters for training")

        logger.info(f"ModelTrainer initialized with use_optuna={use_optuna}")
        logger.debug(f"Initial parameters: {json.dumps(self.params, indent=2)}")

    @staticmethod
    def _get_default_params() -> Dict[str, Any]:
        '''
        Get default XGBoost parameters.

        Returns:
            Dictionary of default parameters
        '''
        return {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'cpu',
            'n_jobs': -1  # Use all CPU cores
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
              early_stopping_rounds: Optional[int] = None,
              eval_metric: Optional[str] = None) -> xgb.XGBRegressor:
        '''
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target values
            X_val: Validation features (optional)
            y_val: Validation target values (optional)
            eval_set: Evaluation set list for XGBoost (optional)
            early_stopping_rounds: Early stopping rounds for XGBoost (optional)
            eval_metric: Evaluation metric for XGBoost (optional)

        Returns:
            Trained XGBRegressor model

        Raises:
            Exception: If training fails
        '''
        logger.info("Starting model training")
        logger.info(f"Training data shape: {X_train.shape}")
        if eval_set:
            logger.info(f"Evaluation set provided with {len(eval_set)} dataset(s)")
        elif X_val is not None:
            logger.info(f"Validation data shape: {X_val.shape}")

        try:
            # Create model with current parameters
            self.model = xgb.XGBRegressor(**self.params)

            # Prepare evaluation set if validation data provided
            eval_set_to_use = eval_set
            if eval_set_to_use is None and X_val is not None and y_val is not None:
                eval_set_to_use = [(X_val, y_val)]

            if early_stopping_rounds is not None and not eval_set_to_use:
                logger.warning("early_stopping_rounds provided without eval_set; early stopping will be ignored")

            # Train model
            start_time = time.time()
            fit_kwargs = {'verbose': False}
            if eval_set_to_use:
                fit_kwargs['eval_set'] = eval_set_to_use
            if early_stopping_rounds is not None and eval_set_to_use:
                fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
            if eval_metric is not None:
                fit_kwargs['eval_metric'] = eval_metric

            self.model.fit(X_train, y_train, **fit_kwargs)

            training_time = time.time() - start_time
            logger.info(f"Model training completed in {training_time:.2f} seconds")

            # Log training completion
            self.training_history.append({
                'timestamp': pd.Timestamp.now().isoformat(),
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'training_time': training_time,
                'has_validation': bool(eval_set_to_use)
            })

            # Log feature importance summary
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = dict(zip(X_train.columns, self.model.feature_importances_))
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                logger.info(f"Top 5 most important features:")
                for feat, imp in top_features[:5]:
                    logger.info(f"  {feat}: {imp:.4f}")

            return self.model

        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5,
                       scoring: str = 'neg_root_mean_squared_error') -> Dict[str, Any]:
        '''
        Perform k-fold cross-validation.

        Args:
            X: Features DataFrame
            y: Target Series
            cv: Number of folds
            scoring: Scoring metric (default: negative RMSE)

        Returns:
            Dictionary with cross-validation results
        '''
        logger.info(f"Starting {cv}-fold cross-validation")

        if self.model is None:
            self.model = xgb.XGBRegressor(**self.params)

        # Define scoring function
        if scoring == 'neg_root_mean_squared_error':
            scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
        else:
            scorer = scoring

        # Perform cross-validation
        # Use n_jobs=1 for GPU to avoid resource conflicts, n_jobs=-1 for CPU
        cv_n_jobs = 1 if self.params.get('device') == 'cuda' else -1
        start_time = time.time()
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scorer,
                                   n_jobs=cv_n_jobs, verbose=0)
        cv_time = time.time() - start_time

        # Calculate metrics
        results = {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'max_cv_score': np.max(cv_scores),
            'min_cv_score': np.min(cv_scores),
            'cv_time': cv_time,
            'n_folds': cv
        }

        logger.info(f"Cross-validation completed in {cv_time:.2f} seconds")
        logger.info(f"CV scores: {cv_scores}")
        logger.info(f"Mean CV score: {results['mean_cv_score']:.4f} (+/- {results['std_cv_score']:.4f})")

        return results

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                                 cv: int = 5, n_trials: Optional[int] = None) -> Dict[str, Any]:
        '''
        Optimize hyperparameters using Optuna.

        Args:
            X: Features DataFrame
            y: Target Series
            cv: Number of folds for cross-validation
            n_trials: Number of optimization trials (uses instance default if None)

        Returns:
            Dictionary with optimization results

        Raises:
            ImportError: If optuna is not installed
        '''
        if not self.use_optuna:
            logger.warning("Optuna optimization not enabled during initialization")
            return {}

        try:
            import optuna
        except ImportError:
            error_msg = "Optuna is not installed. Install with: pip install optuna"
            logger.error(error_msg)
            raise ImportError(error_msg)

        n_trials = n_trials or self.n_trials

        logger.info(f"Starting Optuna hyperparameter optimization with {n_trials} trials")
        logger.info(f"Optimization timeout: {self.optuna_timeout} seconds")

        # Create logs directory if it doesn't exist
        Path('logs').mkdir(parents=True, exist_ok=True)

        # Define objective function for Optuna
        def objective(trial):
            # Stage 1: Mild regularization to reduce overfitting
            # Target: Reduce test COV from 0.134 to < 0.10
            # Strategy: Slightly reduce model capacity, moderate regularization
            params = {
                'objective': 'reg:squarederror',

                # max_depth: MILDLY REDUCED to address overfitting (RMSE ratio = 1.89)
                # Original: 4-9  →  Stage 1: 3-6
                'max_depth': trial.suggest_int('max_depth', 3, 6),

                # learning_rate: LOWERED for more stable convergence
                # Original: 0.01-0.15  →  Stage 1: 0.03-0.12
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.12, log=True),

                # n_estimators: INCREASED to compensate for lower learning rate
                # Original: 300-1000  →  Stage 1: 400-1200
                'n_estimators': trial.suggest_int('n_estimators', 400, 1200),

                # subsample: MODERATELY REDUCED to increase randomness
                # Original: 0.7-0.95  →  Stage 1: 0.6-0.9
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),

                # colsample_bytree: MODERATELY REDUCED for feature sampling
                # Original: 0.7-0.95  →  Stage 1: 0.6-0.9
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),

                # min_child_weight: MILDLY INCREASED to prevent noise fitting
                # Original: 2-15  →  Stage 1: 5-20
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),

                # reg_alpha: MODERATELY INCREASED L1 regularization
                # Original: 0.05-1.5  →  Stage 1: 0.1-2.0
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0, log=True),

                # reg_lambda: MAINTAINED L2 regularization
                # Original: 0.5-5.0  →  Stage 1: 0.5-5.0
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),

                # gamma: MAINTAINED minimum loss reduction
                # Original: 0.01-0.3  →  Stage 1: 0.01-0.3
                'gamma': trial.suggest_float('gamma', 0.01, 0.3),

                # Fixed parameters
                'random_state': self.params.get('random_state', 42),
                'tree_method': 'hist',
                'device': self.params.get('device', 'cpu'),
                'n_jobs': self.params.get('n_jobs', -1)
            }

            # Create and evaluate model
            model = xgb.XGBRegressor(**params)

            # Use cross-validation for evaluation
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.params.get('random_state', 42))
            scores = []

            for train_idx, val_idx in kf.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

                model_fit = xgb.XGBRegressor(**params)
                model_fit.fit(X_train_cv, y_train_cv, verbose=False)

                # Predict and calculate RMSE
                y_pred = model_fit.predict(X_val_cv)
                rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
                scores.append(rmse)

            return np.mean(scores)

        # Create Optuna study with persistent storage
        study = optuna.create_study(
            direction='minimize',
            study_name='xgboost_optimization',
            storage='sqlite:///logs/optuna_study.db',
            load_if_exists=True
        )

        # Run optimization
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, timeout=self.optuna_timeout)
        opt_time = time.time() - start_time

        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        best_trial = study.best_trial

        logger.info(f"Optuna optimization completed in {opt_time:.2f} seconds")
        logger.info(f"Best RMSE: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Save best parameters to file
        save_best_params(
            best_params=best_params,
            best_score=best_score,
            trial_number=best_trial.number,
            n_trials=len(study.trials)
        )

        # Update model parameters with best found parameters
        self.params.update(best_params)
        logger.info(f"Updated model parameters with best parameters")

        # Return optimization results
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(study.trials),
            'optimization_time': opt_time,
            'study': study
        }

        return results

    def get_model_info(self) -> Dict[str, Any]:
        '''
        Get information about the trained model.

        Returns:
            Dictionary with model information
        '''
        if self.model is None:
            logger.warning("No model trained yet")
            return {}

        info = {
            'model_type': 'XGBRegressor',
            'trained': self.model is not None,
            'parameters': self.params,
            'n_features': getattr(self.model, 'n_features_in_', None),
            'feature_names': getattr(self.model, 'feature_names_in_', None),
            'training_history': self.training_history
        }

        return info

    def save_training_history(self, output_path: str) -> None:
        '''
        Save training history to JSON file.

        Args:
            output_path: Path to save training history
        '''
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)

            logger.info(f"Training history saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save training history: {str(e)}")

    def __str__(self) -> str:
        '''
        String representation of ModelTrainer.
        '''
        info = self.get_model_info()
        if not info:
            return "ModelTrainer (no model trained)"

        return f"ModelTrainer(XGBRegressor, n_features={info.get('n_features', 'N/A')})"
