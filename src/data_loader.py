"""
Data Loader Module for CFST XGBoost Pipeline

This module handles loading data from CSV files and preparing it for ML processing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Data loader for CFST XGBoost pipeline.

    Loads data from CSV files, validates column structure, and extracts features and target.
    """

    def __init__(self, required_columns: Optional[List[str]] = None):
        """
        Initialize DataLoader.

        Args:
            required_columns: List of column names that must be present in the data
        """
        self.required_columns = required_columns or []
        self.features_df: Optional[pd.DataFrame] = None
        self.target_series: Optional[pd.Series] = None
        self.target_raw: Optional[pd.Series] = None  # Original target before transform
        self.target_transform: Optional[str] = None  # Store transform type
        self.feature_names: List[str] = []
        self.target_name: str = ""

    def load_data(self, file_path: str, target_column: str,
                  target_transform: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data from CSV file with optional target transformation.

        Args:
            file_path: Path to CSV file
            target_column: Name of the target column
            target_transform: Transformation type ('log', 'sqrt', None)

        Returns:
            Tuple of (features_df, target_series_transformed)

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If required columns are missing or target column not found
        """
        logger.info(f"Loading data from {file_path}")

        # Check if file exists
        if not Path(file_path).exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Load CSV file
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            error_msg = f"Failed to load CSV file: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate required columns
        if self.required_columns:
            missing_columns = set(self.required_columns) - set(df.columns)
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Check if target column exists
        if target_column not in df.columns:
            error_msg = f"Target column '{target_column}' not found in data"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Separate features and target
        self.target_name = target_column
        target_raw = df[target_column].copy()
        self.features_df = df.drop(columns=[target_column]).copy()
        self.feature_names = self.features_df.columns.tolist()

        # Store original target values (for inverse transform and evaluation)
        self.target_raw = target_raw

        # Apply target transformation if specified
        self.target_transform = target_transform
        if target_transform == 'log':
            self.target_series = np.log(target_raw)
            logger.info(f"Applied log transform to target: {target_column}")
            logger.info(f"  Original range: [{target_raw.min():.2f}, {target_raw.max():.2f}]")
            logger.info(f"  Transformed range: [{self.target_series.min():.4f}, {self.target_series.max():.4f}]")
        elif target_transform == 'sqrt':
            self.target_series = np.sqrt(target_raw)
            logger.info(f"Applied sqrt transform to target: {target_column}")
            logger.info(f"  Original range: [{target_raw.min():.2f}, {target_raw.max():.2f}]")
            logger.info(f"  Transformed range: [{self.target_series.min():.4f}, {self.target_series.max():.4f}]")
        else:
            self.target_series = target_raw

        logger.info(f"Data split into {len(self.feature_names)} features and 1 target")
        logger.info(f"Target column: {self.target_name}")
        logger.info(f"Feature names: {self.feature_names[:10]}{'...' if len(self.feature_names) > 10 else ''}")

        return self.features_df, self.target_series

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature column names.

        Returns:
            List of feature names
        """
        if self.feature_names is None:
            logger.warning("No feature names available. Load data first.")
            return []
        return self.feature_names

    def get_target_name(self) -> str:
        """
        Get the target column name.

        Returns:
            Target column name
        """
        if not self.target_name:
            logger.warning("No target name available. Load data first.")
            return ""
        return self.target_name

    def validate_data(self) -> bool:
        """
        Validate loaded data for basic quality checks.

        Returns:
            True if data passes validation, False otherwise
        """
        if self.features_df is None or self.target_series is None:
            logger.error("No data loaded for validation")
            return False

        is_valid = True

        # Check for empty data
        if self.features_df.empty:
            logger.error("Feature dataframe is empty")
            is_valid = False

        if self.target_series.empty:
            logger.error("Target series is empty")
            is_valid = False

        # Check for missing values
        feature_missing = self.features_df.isnull().sum().sum()
        target_missing = self.target_series.isnull().sum()

        if feature_missing > 0:
            logger.warning(f"Found {feature_missing} missing values in features")
            is_valid = False

        if target_missing > 0:
            logger.warning(f"Found {target_missing} missing values in target")
            is_valid = False

        # Log basic statistics
        logger.info(f"Target statistics - Mean: {self.target_series.mean():.2f}, Std: {self.target_series.std():.2f}")
        logger.info(f"Target range: [{self.target_series.min():.2f}, {self.target_series.max():.2f}]")
        logger.info(f"Number of samples: {len(self.features_df)}")

        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning("Data validation failed - see warnings above")

        return is_valid
