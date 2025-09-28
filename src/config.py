"""Configuration management for the financial analysis pipeline."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or "config.yaml"
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults.
        
        Returns:
            Configuration dictionary
        """
        default_config = {
            "database": {
                "path": "financial_data.db"
            },
            "logging": {
                "level": "INFO"
            },
            "data_settings": {
                "historical_period": "5y",
                "min_trading_days_for_sma": 200,
                "fallback_fundamental_data": True,
                "forward_fill_fundamentals": True
            },
            "api_settings": {
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 1
            },
            "markets": {
                "us": {
                    "suffix": "",
                    "exchange": "US"
                },
                "india": {
                    "suffix": ".NS",
                    "exchange": "NSE"
                }
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge user config with defaults
                    return self._merge_configs(default_config, user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_path}: {e}")
                print("Using default configuration.")
        
        return default_config
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user configuration with defaults.
        
        Args:
            default: Default configuration
            user: User configuration
            
        Returns:
            Merged configuration
        """
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @property
    def database_path(self) -> str:
        """Get database path."""
        return self._config["database"]["path"]
    
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self._config["logging"]["level"]
    
    @property
    def historical_period(self) -> str:
        """Get historical period for data fetching."""
        return self._config["data_settings"]["historical_period"]
    
    @property
    def min_trading_days_for_sma(self) -> int:
        """Get minimum trading days required for SMA calculation."""
        return self._config["data_settings"]["min_trading_days_for_sma"]
    
    @property
    def fallback_fundamental_data(self) -> bool:
        """Get whether to use fallback for fundamental data."""
        return self._config["data_settings"]["fallback_fundamental_data"]
    
    @property
    def forward_fill_fundamentals(self) -> bool:
        """Get whether to forward-fill fundamental data."""
        return self._config["data_settings"]["forward_fill_fundamentals"]
    
    @property
    def api_timeout(self) -> int:
        """Get API timeout in seconds."""
        return self._config["api_settings"]["timeout"]
    
    @property
    def api_retry_attempts(self) -> int:
        """Get number of API retry attempts."""
        return self._config["api_settings"]["retry_attempts"]
    
    @property
    def api_retry_delay(self) -> int:
        """Get delay between API retry attempts."""
        return self._config["api_settings"]["retry_delay"]
    
    @property
    def us_market_suffix(self) -> str:
        """Get US market ticker suffix."""
        return self._config["markets"]["us"]["suffix"]
    
    @property
    def india_market_suffix(self) -> str:
        """Get Indian market ticker suffix."""
        return self._config["markets"]["india"]["suffix"]
    
    def get_market_config(self, market: str) -> Dict[str, str]:
        """Get market-specific configuration.
        
        Args:
            market: Market name ('us' or 'india')
            
        Returns:
            Market configuration dictionary
        """
        return self._config["markets"].get(market, {})
    
    def is_valid_ticker(self, ticker: str) -> bool:
        """Check if ticker format is valid for supported markets.
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            True if ticker format is valid
        """
        if not ticker or not isinstance(ticker, str):
            return False
        
        # Check for Indian market suffix
        if ticker.endswith(self.india_market_suffix):
            return True
        
        # Check for US market (no suffix)
        if not any(ticker.endswith(suffix) for suffix in [self.india_market_suffix]):
            return True
        
        return False
    
    def get_market_for_ticker(self, ticker: str) -> str:
        """Determine market for a given ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Market name ('us' or 'india')
        """
        if ticker.endswith(self.india_market_suffix):
            return "india"
        else:
            return "us"
