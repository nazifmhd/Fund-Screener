"""Tests for the data fetcher module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import date, datetime
from decimal import Decimal

from src.data_fetcher import DataFetcher
from src.models import RawAPIData, PriceData, FundamentalData


class TestDataFetcher:
    """Test cases for DataFetcher class."""
    
    def test_data_fetcher_initialization(self):
        """Test data fetcher initialization."""
        fetcher = DataFetcher(timeout=30, retry_attempts=3, retry_delay=1)
        assert fetcher.timeout == 30
        assert fetcher.retry_attempts == 3
        assert fetcher.retry_delay == 1
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_stock_data_success(self, mock_ticker_class, mock_yfinance_data):
        """Test successful stock data fetching."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.info = {"marketCap": 5000000000, "bookValue": 10.0}
        mock_ticker.history.return_value = mock_yfinance_data['history']
        mock_ticker.quarterly_balance_sheet = mock_yfinance_data['quarterly_balance_sheet']
        mock_ticker.balance_sheet = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = DataFetcher()
        
        result = fetcher.fetch_stock_data("TEST", "5y")
        
        assert isinstance(result, RawAPIData)
        assert result.ticker == "TEST"
        assert len(result.price_data) > 0
        assert result.data_source == "yfinance"
        assert result.fetch_timestamp is not None
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_stock_data_invalid_ticker(self, mock_ticker_class):
        """Test data fetching with invalid ticker."""
        # Setup mock to raise exception
        mock_ticker_class.side_effect = Exception("Invalid ticker")
        
        fetcher = DataFetcher()
        
        with pytest.raises(ValueError, match="Failed to fetch ticker"):
            fetcher.fetch_stock_data("INVALID", "5y")
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_stock_data_no_price_data(self, mock_ticker_class):
        """Test data fetching with no price data."""
        # Setup mock with empty history
        mock_ticker = Mock()
        mock_ticker.info = {"marketCap": 5000000000}
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = DataFetcher()
        
        with pytest.raises(ValueError, match="No price data available"):
            fetcher.fetch_stock_data("TEST", "5y")
    
    def test_fetch_stock_data_empty_ticker(self):
        """Test data fetching with empty ticker."""
        fetcher = DataFetcher()
        
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            fetcher.fetch_stock_data("", "5y")
    
    def test_fetch_stock_data_none_ticker(self):
        """Test data fetching with None ticker."""
        fetcher = DataFetcher()
        
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            fetcher.fetch_stock_data(None, "5y")
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_ticker_with_retry_success(self, mock_ticker_class):
        """Test ticker fetching with successful retry."""
        # Setup mock to succeed on first attempt
        mock_ticker = Mock()
        mock_ticker.info = {"marketCap": 5000000000}
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = DataFetcher()
        
        result = fetcher._fetch_ticker_with_retry("TEST")
        
        assert result == mock_ticker
        mock_ticker_class.assert_called_once_with("TEST")
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_fetch_ticker_with_retry_failure(self, mock_ticker_class):
        """Test ticker fetching with retry failure."""
        # Setup mock to always fail
        mock_ticker_class.side_effect = Exception("API Error")
        
        fetcher = DataFetcher(retry_attempts=2, retry_delay=0.01)
        
        with pytest.raises(ValueError, match="Failed to fetch ticker"):
            fetcher._fetch_ticker_with_retry("TEST")
    
    def test_fetch_price_data_success(self, mock_yfinance_data):
        """Test price data fetching."""
        fetcher = DataFetcher()
        
        # Create mock ticker with history
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_yfinance_data['history']
        
        result = fetcher._fetch_price_data(mock_ticker, "5y")
        
        assert len(result) == 3
        assert all(isinstance(price, PriceData) for price in result)
        assert result[0].date == date(2023, 1, 1)
        assert result[0].close == Decimal('102')
    
    def test_fetch_price_data_empty_history(self):
        """Test price data fetching with empty history."""
        fetcher = DataFetcher()
        
        # Create mock ticker with empty history
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        
        result = fetcher._fetch_price_data(mock_ticker, "5y")
        
        assert len(result) == 0
    
    def test_fetch_price_data_invalid_data(self):
        """Test price data fetching with invalid data."""
        fetcher = DataFetcher()
        
        # Create mock ticker with invalid data
        mock_ticker = Mock()
        invalid_data = pd.DataFrame({
            'Open': [100, 'invalid', 104],
            'High': [105, 107, 109],
            'Low': [98, 100, 102],
            'Close': [102, 104, 106],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_ticker.history.return_value = invalid_data
        
        result = fetcher._fetch_price_data(mock_ticker, "5y")
        
        # Should skip invalid records
        assert len(result) == 2  # Only valid records
    
    def test_fetch_fundamental_data_quarterly(self, mock_yfinance_data):
        """Test fundamental data fetching with quarterly data."""
        fetcher = DataFetcher()
        
        # Create mock ticker with quarterly balance sheet
        mock_ticker = Mock()
        mock_ticker.quarterly_balance_sheet = mock_yfinance_data['quarterly_balance_sheet']
        mock_ticker.balance_sheet = pd.DataFrame()
        mock_ticker.info = {"marketCap": 5000000000}
        
        data_quality_issues = []
        result = fetcher._fetch_fundamental_data(mock_ticker, data_quality_issues)
        
        assert result is not None
        assert len(result) == 2
        assert all(isinstance(fund, FundamentalData) for fund in result)
        assert result[0].total_assets == Decimal('1000000000')
    
    def test_fetch_fundamental_data_annual_fallback(self):
        """Test fundamental data fetching with annual fallback."""
        fetcher = DataFetcher()
        
        # Create mock ticker with only annual data
        mock_ticker = Mock()
        mock_ticker.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker.balance_sheet = pd.DataFrame({
            'Total Assets': [1000000000],
            'Total Liab': [400000000],
            'Stockholders Equity': [600000000]
        }, index=pd.date_range('2023-01-01', periods=1))
        mock_ticker.info = {"marketCap": 5000000000}
        
        data_quality_issues = []
        result = fetcher._fetch_fundamental_data(mock_ticker, data_quality_issues)
        
        assert result is not None
        assert len(result) == 1
        assert "Using annual balance sheet data" in data_quality_issues
    
    def test_fetch_fundamental_data_no_data(self):
        """Test fundamental data fetching with no data."""
        fetcher = DataFetcher()
        
        # Create mock ticker with no fundamental data
        mock_ticker = Mock()
        mock_ticker.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker.balance_sheet = pd.DataFrame()
        mock_ticker.info = {"marketCap": 5000000000}
        
        data_quality_issues = []
        result = fetcher._fetch_fundamental_data(mock_ticker, data_quality_issues)
        
        assert result is not None
        assert len(result) == 1  # Synthetic data from info
        assert "Using synthetic fundamental data" in data_quality_issues
    
    def test_fetch_fundamental_data_error(self):
        """Test fundamental data fetching with error."""
        fetcher = DataFetcher()
        
        # Create mock ticker that raises exception
        mock_ticker = Mock()
        mock_ticker.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker.balance_sheet.side_effect = Exception("API Error")
        mock_ticker.info = {}
        
        data_quality_issues = []
        result = fetcher._fetch_fundamental_data(mock_ticker, data_quality_issues)
        
        assert result is None
        assert "Error fetching fundamental data" in data_quality_issues[0]
    
    def test_process_balance_sheet(self, mock_yfinance_data):
        """Test balance sheet processing."""
        fetcher = DataFetcher()
        
        balance_sheet = mock_yfinance_data['quarterly_balance_sheet']
        result = fetcher._process_balance_sheet(balance_sheet, "quarterly")
        
        assert len(result) == 2
        assert all(isinstance(fund, FundamentalData) for fund in result)
        assert result[0].total_assets == Decimal('1000000000')
        assert result[0].total_liabilities == Decimal('400000000')
    
    def test_create_synthetic_fundamental_data(self):
        """Test synthetic fundamental data creation."""
        fetcher = DataFetcher()
        
        info_data = {
            "marketCap": 5000000000,
            "enterpriseValue": 4500000000,
            "bookValue": 10.0,
            "sharesOutstanding": 1000000000
        }
        
        result = fetcher._create_synthetic_fundamental_data(info_data)
        
        assert len(result) == 1
        assert isinstance(result[0], FundamentalData)
        assert result[0].market_cap == Decimal('5000000000')
        assert result[0].book_value_per_share == Decimal('10.0')
    
    def test_create_synthetic_fundamental_data_invalid(self):
        """Test synthetic fundamental data creation with invalid data."""
        fetcher = DataFetcher()
        
        info_data = {
            "marketCap": "invalid",
            "bookValue": None
        }
        
        result = fetcher._create_synthetic_fundamental_data(info_data)
        
        assert len(result) == 1
        assert result[0].market_cap is None
        assert result[0].book_value_per_share is None
    
    def test_safe_decimal_valid_values(self):
        """Test safe decimal conversion with valid values."""
        fetcher = DataFetcher()
        
        assert fetcher._safe_decimal(100) == Decimal('100')
        assert fetcher._safe_decimal(100.5) == Decimal('100.5')
        assert fetcher._safe_decimal("100.5") == Decimal('100.5')
        assert fetcher._safe_decimal(None) is None
        assert fetcher._safe_decimal("invalid") is None
    
    def test_safe_decimal_invalid_values(self):
        """Test safe decimal conversion with invalid values."""
        fetcher = DataFetcher()
        
        assert fetcher._safe_decimal("not_a_number") is None
        assert fetcher._safe_decimal([]) is None
        assert fetcher._safe_decimal({}) is None
