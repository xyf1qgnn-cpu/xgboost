"""
Visualizer Module for CFST XGBoost Pipeline

This module handles data visualization including prediction plots and feature importance charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set style for professional-looking plots
plt.style.use('default')
sns.set_palette("husl")


def plot_predictions_scatter(y_true: pd.Series, y_pred: np.ndarray,
                           title: str = "Predictions vs Actual Values",
                           save_path: Optional[str] = None,
                           r2_score: Optional[float] = None) -> None:
    """
    Generate scatter plot of predictions vs actual values.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
        save_path: Path to save the plot (optional)
        r2_score: R² score to include in title (optional)

    Raises:
        Exception: If plotting fails
    """
    logger.info("Generating predictions vs actual scatter plot")

    try:
        # Create figure
        plt.figure(figsize=(10, 8))

        # Convert to numpy arrays
        y_true_array = np.array(y_true).flatten()
        y_pred_array = np.array(y_pred).flatten()

        # Create scatter plot
        plt.scatter(y_true_array, y_pred_array, alpha=0.6, s=50, color='#2E86AB')

        # Add perfect prediction line
        min_val = min(y_true_array.min(), y_pred_array.min())
        max_val = max(y_true_array.max(), y_pred_array.max())
        margin = (max_val - min_val) * 0.05
        plt.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                'r--', lw=2, label='Perfect Prediction', alpha=0.8)

        # Customize plot
        plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')

        # Update title with R² if provided
        if r2_score is not None:
            final_title = f"{title}\nR² = {r2_score:.4f}"
        else:
            final_title = title
        plt.title(final_title, fontsize=14, fontweight='bold', pad=20)

        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scatter plot saved to {save_path}")

        # Show plot
        plt.show()

        logger.info("Scatter plot generation completed")

    except Exception as e:
        error_msg = f"Failed to generate scatter plot: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def plot_feature_importance(model: Any, feature_names: List[str],
                          title: str = "Feature Importance",
                          save_path: Optional[str] = None,
                          top_n: int = 20) -> None:
    """
    Generate feature importance bar plot from XGBoost model.

    Args:
        model: Trained XGBoost model with feature_importances_ attribute
        feature_names: List of feature names
        title: Plot title
        save_path: Path to save the plot (optional)
        top_n: Number of top features to display

    Raises:
        Exception: If model doesn't have feature_importances_ or plotting fails
    """
    logger.info("Generating feature importance plot")

    try:
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            error_msg = "Model does not have feature_importances_ attribute"
            logger.error(error_msg)
            raise AttributeError(error_msg)

        # Get feature importances
        importances = model.feature_importances_

        if len(importances) != len(feature_names):
            error_msg = f"Feature count mismatch: {len(importances)} importances vs {len(feature_names)} names"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Create DataFrame for easier handling
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)

        # Select top N features
        if len(feature_importance_df) > top_n:
            feature_importance_df = feature_importance_df.tail(top_n)
            title += f" (Top {top_n})"

        # Create figure
        plt.figure(figsize=(12, max(8, len(feature_importance_df) * 0.3)))

        # Create horizontal bar plot
        bars = plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'],
                       color='#2E86AB', alpha=0.8)

        # Customize plot
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontsize=9)

        plt.grid(True, alpha=0.3, axis='x')

        # Adjust layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        # Show plot
        plt.show()

        logger.info("Feature importance plot generation completed")

    except Exception as e:
        error_msg = f"Failed to generate feature importance plot: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def print_feature_importance_ranking(model: Any, feature_names: List[str],
                                   save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Print and optionally save feature importance ranking.

    Args:
        model: Trained XGBoost model with feature_importances_ attribute
        feature_names: List of feature names
        save_path: Path to save the ranking (optional)

    Returns:
        DataFrame with feature rankings

    Raises:
        Exception: If model doesn't have feature_importances_
    """
    logger.info("Generating feature importance ranking")

    try:
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            error_msg = "Model does not have feature_importances_ attribute"
            logger.error(error_msg)
            raise AttributeError(error_msg)

        # Get feature importances
        importances = model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Sort by importance (descending)
        importance_df = importance_df.sort_values('importance', ascending=False)

        # Add rank column
        importance_df['rank'] = range(1, len(importance_df) + 1)

        # Reorder columns
        importance_df = importance_df[['rank', 'feature', 'importance']]

        # Print to console
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE RANKING")
        print("="*80)
        print(importance_df.to_string(index=False))
        print("="*80 + "\n")

        # Save to file if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Save as CSV
            csv_path = save_path if save_path.endswith('.csv') else save_path + '.csv'
            importance_df.to_csv(csv_path, index=False)

            # Also save as formatted text
            txt_path = save_path.replace('.csv', '.txt') if save_path.endswith('.csv') else save_path + '.txt'
            with open(txt_path, 'w') as f:
                f.write("FEATURE IMPORTANCE RANKING\n")
                f.write("="*80 + "\n")
                f.write(importance_df.to_string(index=False))
                f.write("\n" + "="*80 + "\n")

            logger.info(f"Feature importance ranking saved to {csv_path} and {txt_path}")

        logger.info("Feature importance ranking generation completed")

        return importance_df

    except Exception as e:
        error_msg = f"Failed to generate feature importance ranking: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray,
                  title: str = "Residual Plot",
                  save_path: Optional[str] = None) -> None:
    """
    Generate residual plot (errors vs predicted values).

    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
        save_path: Path to save the plot (optional)

    Raises:
        Exception: If plotting fails
    """
    logger.info("Generating residual plot")

    try:
        # Calculate residuals
        residuals = np.array(y_true).flatten() - np.array(y_pred).flatten()

        # Create figure
        plt.figure(figsize=(10, 6))

        # Create scatter plot of residuals
        plt.scatter(y_pred, residuals, alpha=0.6, s=50, color='#A23B72')

        # Add horizontal line at y=0
        plt.axhline(y=0, color='r', linestyle='--', lw=2, alpha=0.8)

        # Customize plot
        plt.xlabel('Predicted Values', fontsize=12, fontweight='bold')
        plt.ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")

        # Show plot
        plt.show()

        logger.info("Residual plot generation completed")

    except Exception as e:
        error_msg = f"Failed to generate residual plot: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def plot_error_distribution(y_true: pd.Series, y_pred: np.ndarray,
                           title: str = "Error Distribution",
                           save_path: Optional[str] = None) -> None:
    """
    Generate error distribution histogram.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
        save_path: Path to save the plot (optional)

    Raises:
        Exception: If plotting fails
    """
    logger.info("Generating error distribution plot")

    try:
        # Calculate errors
        errors = np.array(y_true).flatten() - np.array(y_pred).flatten()

        # Create figure
        plt.figure(figsize=(10, 6))

        # Create histogram
        plt.hist(errors, bins=50, alpha=0.7, color='#F18F01', edgecolor='black')

        # Customize plot
        plt.xlabel('Prediction Error', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y')

        # Add statistics text
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        plt.text(0.05, 0.95, f'Mean Error: {mean_error:.4f}\nStd Error: {std_error:.4f}',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Adjust layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error distribution plot saved to {save_path}")

        # Show plot
        plt.show()

        logger.info("Error distribution plot generation completed")

    except Exception as e:
        error_msg = f"Failed to generate error distribution plot: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def create_evaluation_dashboard(y_true: pd.Series, y_pred: np.ndarray,
                               model: Any, feature_names: List[str],
                               output_dir: str, model_name: str = "model") -> None:
    """
    Create a comprehensive evaluation dashboard with multiple plots.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        model: Trained model
        feature_names: List of feature names
        output_dir: Directory to save all plots
        model_name: Name of the model

    Raises:
        Exception: If dashboard creation fails
    """
    logger.info(f"Creating evaluation dashboard for {model_name}")

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate R² for scatter plot title
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)

        # Generate all plots
        plot_predictions_scatter(
            y_true, y_pred,
            title=f"{model_name} - Predictions vs Actual",
            save_path=str(output_path / f"{model_name}_predictions_scatter.png"),
            r2_score=r2
        )

        plot_residuals(
            y_true, y_pred,
            title=f"{model_name} - Residual Plot",
            save_path=str(output_path / f"{model_name}_residuals.png")
        )

        plot_error_distribution(
            y_true, y_pred,
            title=f"{model_name} - Error Distribution",
            save_path=str(output_path / f"{model_name}_error_distribution.png")
        )

        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(
                model, feature_names,
                title=f"{model_name} - Feature Importance",
                save_path=str(output_path / f"{model_name}_feature_importance.png")
            )

            print_feature_importance_ranking(
                model, feature_names,
                save_path=str(output_path / f"{model_name}_feature_ranking")
            )

        logger.info(f"Evaluation dashboard created in {output_dir}")

    except Exception as e:
        error_msg = f"Failed to create evaluation dashboard: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
