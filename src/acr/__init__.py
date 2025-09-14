"""Agent-Based Credit Risk (ACR) - 信贷风控代理模型."""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from acr.config.loader import load_config
from acr.config.schema import Config

__all__ = ["Config", "load_config"]
