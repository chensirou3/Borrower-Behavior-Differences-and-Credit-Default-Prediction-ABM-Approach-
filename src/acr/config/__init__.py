"""Configuration management for ACR."""

from acr.config.loader import load_config
from acr.config.schema import Config

__all__ = ["Config", "load_config"]
