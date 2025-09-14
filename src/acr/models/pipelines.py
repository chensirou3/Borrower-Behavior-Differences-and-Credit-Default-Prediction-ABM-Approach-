"""Model training pipelines."""

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

from acr.config.schema import Config, AlgorithmConfig


def train_models(
    X_train_baseline: pd.DataFrame,
    X_train_augmented: pd.DataFrame, 
    y_train: pd.Series,
    X_test_baseline: pd.DataFrame,
    X_test_augmented: pd.DataFrame,
    y_test: pd.Series,
    config: Config
) -> Dict[str, Any]:
    """Train all configured models on baseline and augmented features.
    
    Args:
        X_train_baseline: Training baseline features
        X_train_augmented: Training augmented features
        y_train: Training targets
        X_test_baseline: Test baseline features
        X_test_augmented: Test augmented features
        y_test: Test targets
        config: Configuration object
        
    Returns:
        Dictionary with trained models and predictions
    """
    results = {}
    
    # Train each algorithm on both feature sets
    for algo_config in config.modeling.algorithms:
        algo_name = algo_config.name
        
        # Baseline features
        model_baseline = _train_single_model(
            X_train_baseline, y_train, algo_config
        )
        pred_baseline = model_baseline.predict_proba(X_test_baseline)[:, 1]
        
        results[f"{algo_name}_baseline"] = {
            'model': model_baseline,
            'predictions': pred_baseline,
            'feature_set': 'baseline'
        }
        
        # Augmented features
        model_augmented = _train_single_model(
            X_train_augmented, y_train, algo_config
        )
        pred_augmented = model_augmented.predict_proba(X_test_augmented)[:, 1]
        
        results[f"{algo_name}_augmented"] = {
            'model': model_augmented,
            'predictions': pred_augmented,
            'feature_set': 'augmented'
        }
    
    return results


def _train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algo_config: AlgorithmConfig
) -> Any:
    """Train a single model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        algo_config: Algorithm configuration
        
    Returns:
        Trained model
    """
    if algo_config.name == "logistic":
        return _train_logistic(X_train, y_train, algo_config)
    elif algo_config.name == "xgboost":
        return _train_xgboost(X_train, y_train, algo_config)
    else:
        raise ValueError(f"Unknown algorithm: {algo_config.name}")


def _train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algo_config: AlgorithmConfig
) -> Any:
    """Train logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        algo_config: Algorithm configuration
        
    Returns:
        Trained logistic regression model
    """
    # Base logistic regression
    base_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        **algo_config.params
    )
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', base_model)
    ])
    
    # Apply calibration if requested
    if algo_config.calibrate == "platt":
        model = CalibratedClassifierCV(pipeline, method='sigmoid', cv=3)
    elif algo_config.calibrate == "isotonic":
        model = CalibratedClassifierCV(pipeline, method='isotonic', cv=3)
    else:
        model = pipeline
    
    # Fit model
    model.fit(X_train, y_train)
    
    return model


def _train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algo_config: AlgorithmConfig
) -> Any:
    """Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        algo_config: Algorithm configuration
        
    Returns:
        Trained XGBoost model
    """
    # Default XGBoost parameters
    default_params = {
        'n_estimators': 200,
        'max_depth': 3,
        'learning_rate': 0.08,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Update with config parameters
    params = {**default_params, **algo_config.params}
    
    # Base XGBoost classifier
    base_model = xgb.XGBClassifier(**params)
    
    # Apply calibration if requested
    if algo_config.calibrate == "platt":
        model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    elif algo_config.calibrate == "isotonic":
        model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    else:
        model = base_model
    
    # Fit model
    model.fit(X_train, y_train)
    
    return model


def get_feature_importance(
    models: Dict[str, Any],
    feature_names: Dict[str, list[str]]
) -> Dict[str, Dict[str, float]]:
    """Extract feature importance from trained models.
    
    Args:
        models: Dictionary of trained models
        feature_names: Dictionary mapping model names to feature names
        
    Returns:
        Dictionary of feature importance scores
    """
    importance_dict = {}
    
    for model_name, model_info in models.items():
        model = model_info['model']
        feature_set = model_info['feature_set']
        
        if feature_set in feature_names:
            features = feature_names[feature_set]
        else:
            continue
        
        # Extract importance based on model type
        if hasattr(model, 'feature_importances_'):
            # XGBoost or tree-based
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0])
        elif hasattr(model, 'calibrated_classifiers_'):
            # Calibrated models
            base_model = model.calibrated_classifiers_[0]
            if hasattr(base_model, 'feature_importances_'):
                importance = base_model.feature_importances_
            elif hasattr(base_model, 'coef_'):
                importance = np.abs(base_model.coef_[0])
            else:
                continue
        else:
            continue
        
        # Create importance dictionary
        if len(importance) == len(features):
            importance_dict[model_name] = dict(zip(features, importance))
    
    return importance_dict


def predict_with_models(
    models: Dict[str, Any],
    X_baseline: pd.DataFrame,
    X_augmented: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """Generate predictions with all models.
    
    Args:
        models: Dictionary of trained models
        X_baseline: Baseline features
        X_augmented: Augmented features
        
    Returns:
        Dictionary of prediction arrays
    """
    predictions = {}
    
    for model_name, model_info in models.items():
        model = model_info['model']
        feature_set = model_info['feature_set']
        
        # Select appropriate feature set
        if feature_set == 'baseline':
            X = X_baseline
        elif feature_set == 'augmented':
            X = X_augmented
        else:
            continue
        
        # Generate predictions
        pred_proba = model.predict_proba(X)[:, 1]
        predictions[model_name] = pred_proba
    
    return predictions
