"""Configuration loading and merging utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

from acr.config.schema import Config


def _set_nested_value(data: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a nested value in a dictionary using dot notation.
    
    Args:
        data: Dictionary to modify
        key_path: Dot-separated key path (e.g., 'a.b.c')
        value: Value to set
    """
    keys = key_path.split('.')
    current = data
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def _parse_override_value(value_str: str) -> Any:
    """Parse a string value to appropriate Python type.
    
    Args:
        value_str: String representation of the value
        
    Returns:
        Parsed value with appropriate type
    """
    # Try to parse as JSON first (handles lists, dicts, booleans, numbers, null)
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        pass
    
    # Return as string if JSON parsing fails
    return value_str


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def merge_configs(base_config: Dict[str, Any], *override_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        *override_configs: Additional configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for override_config in override_configs:
        _deep_merge(result, override_config)
    
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Deep merge override dictionary into base dictionary (in-place).
    
    Args:
        base: Base dictionary to merge into
        override: Override dictionary to merge from
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply command-line overrides to configuration.
    
    Args:
        config: Base configuration dictionary
        overrides: List of override strings in format "key.path=value"
        
    Returns:
        Configuration with overrides applied
        
    Raises:
        ValueError: If override format is invalid
    """
    result = config.copy()
    
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Expected 'key.path=value'")
        
        key_path, value_str = override.split('=', 1)
        value = _parse_override_value(value_str)
        _set_nested_value(result, key_path, value)
    
    return result


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
    validate: bool = True
) -> Config:
    """Load and validate configuration.
    
    Args:
        config_path: Path to YAML configuration file
        overrides: List of override strings in format "key.path=value"
        validate: Whether to validate the configuration
        
    Returns:
        Validated configuration object
        
    Raises:
        ValidationError: If configuration validation fails
        FileNotFoundError: If config file doesn't exist
        ValueError: If override format is invalid
    """
    # Start with empty config
    config_dict: Dict[str, Any] = {}
    
    # Load from file if provided
    if config_path is not None:
        config_dict = load_yaml_config(config_path)
    
    # Apply overrides if provided
    if overrides:
        config_dict = apply_overrides(config_dict, overrides)
    
    # Validate and return
    if validate:
        try:
            return Config(**config_dict)
        except ValidationError as e:
            raise ValidationError(f"Configuration validation failed: {e}")
    
    return Config(**config_dict)


def save_config(config: Config, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        output_path: Path where to save the configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary and save as YAML
    config_dict = config.model_dump()
    
    # Convert numpy types to Python types for YAML serialization
    config_dict = _convert_numpy_types(config_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python types for JSON/YAML serialization."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
