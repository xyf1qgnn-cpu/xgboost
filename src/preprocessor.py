"""
Preprocessor Module for CFST XGBoost Pipeline

This module handles data preprocessing including column dropping and data cleaning.
"""

import pandas as pd
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Data preprocessor for CFST XGBoost pipeline.

    Handles column removal (especially geometric parameters) and basic data cleaning.
    Modeled after scikit-learn's Transformer interface for consistency.
    """

    def __init__(
        self,
        columns_to_drop: List[str],
        outlier_method: str = "iqr",
        z_threshold: float = 3.0,
        outlier_strategy: str = "none",
        iqr_factor: float = 1.5,
    ):
        """
        Initialize Preprocessor.

        Args:
            columns_to_drop: List of column names to drop from the data
            outlier_method: Outlier detection method ("iqr" or "z_score")
            z_threshold: Z-score threshold for outlier detection
            outlier_strategy: Strategy to handle outliers ("none", "drop_outliers", "clip_outliers")
            iqr_factor: IQR multiplier for bounds when using IQR method
        """
        self.columns_to_drop = columns_to_drop
        self.remaining_features: List[str] = []
        self.is_fitted = False
        self.outlier_method = self._normalize_outlier_method(outlier_method)
        self.z_threshold = z_threshold
        self.outlier_strategy = self._normalize_outlier_strategy(outlier_strategy)
        self.iqr_factor = iqr_factor
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.target_outlier_bounds: Optional[Tuple[float, float]] = None
        logger.info(f"Preprocessor initialized with {len(columns_to_drop)} columns to drop: {columns_to_drop}")
        logger.info(
            "Outlier handling configured",
            extra={
                "method": self.outlier_method,
                "strategy": self.outlier_strategy,
                "z_threshold": self.z_threshold,
                "iqr_factor": self.iqr_factor,
            },
        )

    @staticmethod
    def _normalize_outlier_method(method: str) -> str:
        normalized = (method or "iqr").strip().lower()
        if normalized in {"z", "zscore", "z-score"}:
            return "z_score"
        if normalized in {"iqr"}:
            return "iqr"
        raise ValueError(f"Unsupported outlier_method: {method}")

    @staticmethod
    def _normalize_outlier_strategy(strategy: str) -> str:
        normalized = (strategy or "none").strip().lower()
        if normalized in {"none", "disabled", "off"}:
            return "none"
        if normalized in {"drop", "drop_outliers", "drop-outliers"}:
            return "drop_outliers"
        if normalized in {"clip", "clip_outliers", "clip-outliers"}:
            return "clip_outliers"
        raise ValueError(f"Unsupported outlier_strategy: {strategy}")

    def _get_numeric_columns(self, X: pd.DataFrame) -> List[str]:
        return X.select_dtypes(include=[np.number]).columns.tolist()

    def _calculate_bounds(self, series: pd.Series) -> Tuple[float, float]:
        if self.outlier_method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr
            return lower, upper
        mean = series.mean()
        std = series.std()
        if pd.isna(std) or std == 0:
            return mean, mean
        lower = mean - self.z_threshold * std
        upper = mean + self.z_threshold * std
        return lower, upper

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Preprocessor":
        """
        Fit the preprocessor to the data.

        Identifies which columns will remain after dropping specified columns.

        Args:
            X: Input features DataFrame

        Returns:
            self: Fitted preprocessor instance

        Raises:
            ValueError: If required columns to drop are not found in data
        """
        logger.info("Fitting preprocessor to data")

        # Check if columns to drop exist in the data
        missing_columns = set(self.columns_to_drop) - set(X.columns)
        if missing_columns:
            error_msg = f"Columns to drop not found in data: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Determine remaining features
        self.remaining_features = [col for col in X.columns if col not in self.columns_to_drop]

        logger.info(f"Original features: {len(X.columns)}")
        logger.info(f"Columns to drop: {len(self.columns_to_drop)}")
        logger.info(f"Remaining features: {len(self.remaining_features)}")
        logger.debug(f"Remaining feature names: {self.remaining_features}")

        # Fit outlier bounds on training data
        self.outlier_bounds = {}
        self.target_outlier_bounds = None
        if self.outlier_strategy != "none":
            numeric_columns = self._get_numeric_columns(X[self.remaining_features])
            for column in numeric_columns:
                self.outlier_bounds[column] = self._calculate_bounds(X[column])

            if y is not None:
                if pd.api.types.is_numeric_dtype(y):
                    self.target_outlier_bounds = self._calculate_bounds(y)
                else:
                    logger.warning("Target provided for outlier handling is not numeric; skipping target bounds.")

        self.is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]]:
        """
        Transform the data by dropping specified columns.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with specified columns removed

        Raises:
            ValueError: If preprocessor is not fitted or if data is missing expected columns
        """
        if not self.is_fitted:
            error_msg = "Preprocessor must be fitted before transform. Call fit() first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Transforming data with shape {X.shape}")

        # Check if all remaining features are present
        missing_features = set(self.remaining_features) - set(X.columns)
        if missing_features:
            error_msg = f"Missing expected features: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Drop columns and return remaining features
        X_transformed = X[self.remaining_features].copy()
        y_transformed = y.copy() if y is not None else None

        outlier_stats: Dict[str, Any] = {
            "strategy": self.outlier_strategy,
            "method": self.outlier_method,
            "n_samples_before": len(X_transformed),
            "n_samples_after": len(X_transformed),
            "n_dropped": 0,
            "feature_outlier_counts": {},
            "target_outlier_count": 0,
        }

        if self.outlier_strategy != "none" and (self.outlier_bounds or self.target_outlier_bounds is not None):
            if self.outlier_strategy == "drop_outliers":
                mask = pd.Series(True, index=X_transformed.index)
                for column, (lower, upper) in self.outlier_bounds.items():
                    column_mask = X_transformed[column].between(lower, upper, inclusive="both")
                    outlier_count = (~column_mask).sum()
                    outlier_stats["feature_outlier_counts"][column] = int(outlier_count)
                    mask &= column_mask

                if y_transformed is not None and self.target_outlier_bounds is not None:
                    lower, upper = self.target_outlier_bounds
                    target_mask = y_transformed.between(lower, upper, inclusive="both")
                    outlier_stats["target_outlier_count"] = int((~target_mask).sum())
                    mask &= target_mask

                X_transformed = X_transformed.loc[mask].copy()
                if y_transformed is not None:
                    y_transformed = y_transformed.loc[mask].copy()

                outlier_stats["n_samples_after"] = len(X_transformed)
                outlier_stats["n_dropped"] = outlier_stats["n_samples_before"] - outlier_stats["n_samples_after"]

            elif self.outlier_strategy == "clip_outliers":
                for column, (lower, upper) in self.outlier_bounds.items():
                    if column in X_transformed.columns:
                        outlier_count = (~X_transformed[column].between(lower, upper, inclusive="both")).sum()
                        outlier_stats["feature_outlier_counts"][column] = int(outlier_count)
                        X_transformed[column] = X_transformed[column].clip(lower=lower, upper=upper)

                if y_transformed is not None and self.target_outlier_bounds is not None:
                    lower, upper = self.target_outlier_bounds
                    outlier_stats["target_outlier_count"] = int(
                        (~y_transformed.between(lower, upper, inclusive="both")).sum()
                    )
                    y_transformed = y_transformed.clip(lower=lower, upper=upper)

        logger.info(f"Transformed data shape: {X_transformed.shape}")
        logger.debug(f"Dropped {len(self.columns_to_drop)} columns: {self.columns_to_drop}")

        if y is not None:
            return X_transformed, y_transformed, outlier_stats

        return X_transformed

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]]:
        """
        Fit the preprocessor and transform the data in one step.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with specified columns removed
        """
        return self.fit(X, y).transform(X, y)

    def get_remaining_features(self) -> List[str]:
        """
        Get the list of features that will remain after dropping.

        Returns:
            List of remaining feature names

        Raises:
            ValueError: If preprocessor is not fitted yet
        """
        if not self.is_fitted:
            error_msg = "Preprocessor must be fitted before getting remaining features. Call fit() first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        return self.remaining_features.copy()

    def get_dropped_columns(self) -> List[str]:
        """
        Get the list of columns that will be dropped.

        Returns:
            List of columns to drop
        """
        return self.columns_to_drop.copy()

    def is_column_dropped(self, column: str) -> bool:
        """
        Check if a specific column will be dropped.

        Args:
            column: Column name to check

        Returns:
            True if column will be dropped, False otherwise
        """
        return column in self.columns_to_drop

    def check_missing_values(self, X: pd.DataFrame) -> dict:
        """
        Check for missing values in the data.

        Args:
            X: DataFrame to check

        Returns:
            Dictionary with column names as keys and missing value counts as values
        """
        missing_counts = X.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]

        if missing_counts.empty:
            logger.info("No missing values found")
        else:
            logger.warning(f"Found missing values:\n{missing_counts}")

        return missing_counts.to_dict()

    def get_feature_stats(self, X: pd.DataFrame) -> dict:
        """
        Get basic statistics for the features.

        Args:
            X: Features DataFrame

        Returns:
            Dictionary with statistics
        """
        stats = {
            "n_samples": len(X),
            "n_features": len(X.columns),
            "feature_names": list(X.columns),
            "missing_values": self.check_missing_values(X),
            "numeric_features": X.select_dtypes(include=[np.number]).columns.tolist(),
        }

        logger.info(f"Feature stats: {len(X.columns)} features, {len(X)} samples")

        return stats
