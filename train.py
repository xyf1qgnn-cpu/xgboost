#!/usr/bin/env python3
"""
Training Script for CFST XGBoost Pipeline

This script executes the complete training pipeline:
1. Load configuration
2. Load data
3. Preprocess data
4. Train XGBoost model
5. Evaluate model
6. Save model and results

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --output models/my_model
"""

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

from src.utils.logger import setup_logger
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator
from src.utils.model_utils import save_model, save_metadata
from src.visualizer import (create_evaluation_dashboard, plot_feature_importance,
                           print_feature_importance_ranking)

logger = setup_logger(__name__)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config_path: str, output_dir: str = None) -> dict:
    """
    Execute complete training pipeline.

    Args:
        config_path: Path to configuration YAML file
        output_dir: Output directory for saving model and results (optional)

    Returns:
        Dictionary with training results

    Raises:
        Exception: If any step in the pipeline fails
    """
    logger.info("=" * 80)
    logger.info("CFST XGBOOST PIPELINE - TRAINING STARTED")
    logger.info("=" * 80)

    try:
        # Step 1: Load configuration
        logger.info("Step 1: Loading configuration...")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")

        # Extract paths and parameters
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        cv_config = config.get('cv', {})
        output_config = config.get('paths', {})

        data_path = data_config.get('file_path')
        target_column = data_config.get('target_column', 'K')
        columns_to_drop = data_config.get('columns_to_drop', [])

        # Set output directory
        if output_dir is None:
            output_dir = output_config.get('output_dir', 'output')

        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Data file: {data_path}")
        logger.info(f"Target column: {target_column}")
        logger.info(f"Columns to drop: {columns_to_drop}")

        # Step 2: Load data
        logger.info("\nStep 2: Loading data...")
        data_loader = DataLoader(required_columns=[target_column])
        features, target = data_loader.load_data(data_path, target_column)
        logger.info(f"Data loaded: {len(features)} samples, {len(features.columns)} features")

        # Step 3: Preprocess data
        logger.info("\nStep 3: Preprocessing data...")
        preprocessor = Preprocessor(columns_to_drop=columns_to_drop)
        features_processed = preprocessor.fit_transform(features)
        logger.info(f"Preprocessing completed: {len(features_processed.columns)} features remaining")

        # Get remaining feature names
        feature_names = preprocessor.get_remaining_features()
        logger.info(f"Remaining features: {feature_names}")

        # Check for missing values
        missing_info = preprocessor.check_missing_values(features_processed)
        if missing_info:
            logger.warning(f"Found missing values: {missing_info}")
        else:
            logger.info("No missing values found")

        # Step 4: Train model
        logger.info("\nStep 4: Training XGBoost model...")

        # Initialize trainer with parameters
        use_optuna = model_config.get('use_optuna', False)
        n_trials = model_config.get('optuna_trials', 100)
        optuna_timeout = model_config.get('optuna_timeout', 3600)

        # Prepare XGBoost parameters
        xgb_params = {
            'objective': model_config.get('objective', 'reg:squarederror'),
            'max_depth': model_config.get('max_depth', 6),
            'learning_rate': model_config.get('learning_rate', 0.1),
            'n_estimators': model_config.get('n_estimators', 200),
            'subsample': model_config.get('subsample', 0.8),
            'colsample_bytree': model_config.get('colsample_bytree', 0.8),
            'random_state': model_config.get('random_state', 42),
            'tree_method': model_config.get('tree_method', 'hist'),
            'device': model_config.get('device', 'cpu'),
            'n_jobs': model_config.get('n_jobs', -1)
        }

        trainer = ModelTrainer(params=xgb_params, use_optuna=use_optuna,
                             n_trials=n_trials, optuna_timeout=optuna_timeout)

        # Train model
        model = trainer.train(features_processed, target)
        logger.info("Model training completed")

        # Optional: Hyperparameter optimization
        if use_optuna:
            logger.info("Starting Optuna hyperparameter optimization...")
            opt_results = trainer.optimize_hyperparameters(
                features_processed, target,
                cv=cv_config.get('n_splits', 5)
            )
            logger.info(f"Optuna optimization completed: {opt_results['n_trials']} trials")

        # Step 5: Cross-validation
        logger.info("\nStep 5: Performing cross-validation...")
        cv_results = trainer.cross_validate(
            features_processed, target,
            cv=cv_config.get('n_folds', 5)
        )
        logger.info(f"Cross-validation RMSE: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score']:.4f})")

        # Step 6: Evaluate model
        logger.info("\nStep 6: Evaluating model...")
        evaluator = Evaluator()

        # Make predictions on training data
        from src.predictor import Predictor
        predictor = Predictor(model, preprocessor, feature_names)
        predictions = predictor.predict(features)

        # Calculate metrics
        evaluation_result = evaluator.calculate_metrics(target, predictions)
        logger.info(f"Model evaluation completed:")
        logger.info(f"  RMSE: {evaluation_result['rmse']:.4f}")
        logger.info(f"  MAE: {evaluation_result['mae']:.4f}")
        logger.info(f"  R²: {evaluation_result['r2']:.4f}")
        if evaluation_result['mape']:
            logger.info(f"  MAPE: {evaluation_result['mape']:.2f}%")

        # Step 7: Create visualizations
        logger.info("\nStep 7: Creating visualizations...")
        output_path = Path(output_dir)
        plots_dir = output_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate prediction scatter plot
        create_evaluation_dashboard(
            target, predictions, model, feature_names,
            str(plots_dir), "xgboost_model"
        )

        logger.info(f"Visualizations saved to {plots_dir}")

        # Step 8: Save model and artifacts
        logger.info("\nStep 8: Saving model and artifacts...")

        # Save model with metadata
        metadata = {
            'config': config,
            'evaluation_metrics': evaluation_result,
            'cross_validation_results': cv_results,
            'feature_names': feature_names,
            'n_samples': len(features),
            'n_features': len(feature_names),
            'training_successful': True
        }

        save_model(
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            output_dir=output_dir,
            metadata=metadata
        )

        # Prepare evaluation report (convert numpy arrays to lists for JSON)
        import numpy as np

        # Convert cv_results to be JSON serializable
        serializable_cv_results = {}
        if isinstance(cv_results, dict):
            for key, value in cv_results.items():
                if hasattr(value, 'tolist'):  # numpy arrays
                    serializable_cv_results[key] = value.tolist()
                elif isinstance(value, dict) and 'cv_scores' in str(value):
                    # Handle nested CV results
                    serializable_cv_results[key] = {}
                    for k2, v2 in value.items():
                        if hasattr(v2, 'tolist'):
                            serializable_cv_results[key][k2] = v2.tolist()
                        else:
                            serializable_cv_results[key][k2] = v2
                else:
                    serializable_cv_results[key] = value

        eval_report_path = output_path / "evaluation_report.json"
        evaluator.save_evaluation_report(
            {
                'model_name': 'xgboost_model',
                'timestamp': pd.Timestamp.now().isoformat(),
                'metrics': evaluation_result,
                'cv_results': serializable_cv_results
            },
            str(eval_report_path)
        )

        logger.info(f"Model and artifacts saved to {output_dir}")

        # Step 9: Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {output_dir}/xgboost_model.pkl")
        logger.info(f"Preprocessor saved to: {output_dir}/preprocessor.pkl")
        logger.info(f"Evaluation report: {eval_report_path}")
        logger.info(f"Plots saved to: {plots_dir}/")
        logger.info("=" * 80)

        return {
            'model': model,
            'preprocessor': preprocessor,
            'metrics': evaluation_result,
            'cv_results': cv_results,
            'feature_names': feature_names,
            'output_dir': output_dir
        }

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description='Train CFST XGBoost Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default configuration
  python train.py

  # Custom configuration file
  python train.py --config config/config.yaml

  # Custom output directory
  python train.py --output my_model_output

  # Custom config and output
  python train.py --config config/config.yaml --output models/cfst_model
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to configuration YAML file (default: config/config.yaml)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for model and results (default: from config)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Check if config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Please check the file path or create a config file using config/config.example.yaml")
        sys.exit(1)

    try:
        # Run training pipeline
        train_model(args.config, args.output)
        logger.info("Training completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Use --verbose for more details")
        if args.verbose:
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
