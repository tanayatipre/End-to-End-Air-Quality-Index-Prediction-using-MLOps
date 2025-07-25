from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    csv_file_path: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    preprocessor_name: str
    train_data_path: Path
    test_data_path: Path
    target_column: str
    numerical_cols: List[str]
    categorical_cols: List[str]
    columns_to_log_transform: List[str]
    columns_to_drop_after_feature_eng: List[str]
    test_size: float

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    params: Dict[str, Any]
    target_column: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str