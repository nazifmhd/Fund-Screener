"""Tests for the data processor module."""

import pytest
import pandas as pd
from decimal import Decimal
from datetime import date

from src.processor import DataProcessor
from src.models import RawAPIData, PriceData, FundamentalData


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = DataProcessor(min_trading_days_for_sma=200, forward_fill_fundamentals=True)
        assert processor.min_trading_days_for_sma == 200
        assert processor.forward_fill_fundamentals is True
    
    def test_create_price_dataframe(self, sample_price_data):
        """Test price data DataFrame creation."""
        processor = DataProcessor()
        df = processor._create_price_dataframe(sample_price_data)
        
        assert len(df) == 3
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert df.index.name == 'date'
        assert df.loc[date(2023, 1, 1), 'close'] == 102.0
    
    def test_process_fundamental_data(self, sample_fundamental_data):
        """Test fundamental data processing."""
        processor = DataProcessor()
        df = processor._process_fundamental_data(sample_fundamental_data, None)
        
        assert df is not None
        assert len(df) == 1
        assert 'total_assets' in df.columns
        assert 'book_value_per_share' in df.columns
        assert df.loc[date(2023, 1, 1), 'total_assets'] == 1000000000.0
    
    def test_process_fundamental_data_none(self):
        """Test fundamental data processing with None input."""
        processor = DataProcessor()
        df = processor._process_fundamental_data(None, None)
        
        assert df is None
    
    def test_merge_price_and_fundamental_data(self, sample_price_data, sample_fundamental_data):
        """Test merging price and fundamental data."""
        processor = DataProcessor()
        
        price_df = processor._create_price_dataframe(sample_price_data)
        fundamental_df = processor._process_fundamental_data(sample_fundamental_data, None)
        
        merged_df = processor._merge_price_and_fundamental_data(price_df, fundamental_df)
        
        assert len(merged_df) == 3
        assert 'total_assets' in merged_df.columns
        assert 'close' in merged_df.columns
    
    def test_merge_price_and_fundamental_data_no_fundamental(self, sample_price_data):
        """Test merging when no fundamental data is available."""
        processor = DataProcessor()
        
        price_df = processor._create_price_dataframe(sample_price_data)
        merged_df = processor._merge_price_and_fundamental_data(price_df, None)
        
        assert len(merged_df) == 3
        assert 'close' in merged_df.columns
        assert 'total_assets' not in merged_df.columns
    
    def test_calculate_sma(self):
        """Test SMA calculation."""
        processor = DataProcessor()
        
        # Create test series
        series = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])
        
        # Test 5-day SMA
        sma_5 = processor._calculate_sma(series, 5)
        
        assert len(sma_5) == len(series)
        assert pd.isna(sma_5.iloc[0:4]).all()  # First 4 should be NaN
        assert not pd.isna(sma_5.iloc[4:]).any()  # Rest should have values
        
        # Check first valid value
        expected_first = series.iloc[0:5].mean()
        assert abs(sma_5.iloc[4] - expected_first) < 0.001
    
    def test_calculate_sma_insufficient_data(self):
        """Test SMA calculation with insufficient data."""
        processor = DataProcessor()
        
        # Create short series
        series = pd.Series([100, 102, 104])
        
        sma_50 = processor._calculate_sma(series, 50)
        
        assert len(sma_50) == len(series)
        assert sma_50.isna().all()  # All should be NaN
    
    def test_calculate_technical_indicators(self, sample_dataframe):
        """Test technical indicators calculation."""
        processor = DataProcessor()
        
        result_df = processor._calculate_technical_indicators(sample_dataframe)
        
        assert 'sma_50' in result_df.columns
        assert 'sma_200' in result_df.columns
        assert 'high_52w' in result_df.columns
        assert 'pct_from_52w_high' in result_df.columns
        
        # Check that SMA columns are properly calculated
        assert not result_df['sma_50'].isna().all()
        assert not result_df['sma_200'].isna().all()
    
    def test_calculate_fundamental_ratios(self, sample_dataframe):
        """Test fundamental ratios calculation."""
        processor = DataProcessor()
        
        # Add some fundamental data
        sample_dataframe['book_value_per_share'] = 10.0
        sample_dataframe['market_cap'] = 1000000000
        sample_dataframe['total_liabilities'] = 400000000
        
        result_df = processor._calculate_fundamental_ratios(sample_dataframe, None)
        
        assert 'price_to_book' in result_df.columns
        assert 'enterprise_value' in result_df.columns
        
        # Check price-to-book calculation
        expected_pbr = sample_dataframe['close'] / sample_dataframe['book_value_per_share']
        pd.testing.assert_series_equal(result_df['price_to_book'], expected_pbr)
    
    def test_dataframe_to_dict_list(self, sample_dataframe):
        """Test DataFrame to dictionary list conversion."""
        processor = DataProcessor()
        
        data_list = processor._dataframe_to_dict_list(sample_dataframe)
        
        assert len(data_list) == len(sample_dataframe)
        assert isinstance(data_list[0], dict)
        assert 'date' in data_list[0]
        assert 'close' in data_list[0]
    
    def test_get_data_quality_summary(self, sample_dataframe):
        """Test data quality summary generation."""
        processor = DataProcessor()
        
        summary = processor.get_data_quality_summary(sample_dataframe)
        
        assert 'total_records' in summary
        assert 'date_range' in summary
        assert 'missing_data' in summary
        assert 'data_completeness' in summary
        
        assert summary['total_records'] == len(sample_dataframe)
        assert summary['date_range']['start'] is not None
        assert summary['date_range']['end'] is not None
    
    def test_process_data_integration(self, sample_raw_data):
        """Test full data processing pipeline."""
        processor = DataProcessor()
        
        processed_data = processor.process_data(sample_raw_data)
        
        assert processed_data.ticker == "TEST"
        assert len(processed_data.data) > 0
        assert 'date' in processed_data.data[0]
        assert 'close' in processed_data.data[0]
        assert processed_data.processing_metadata is not None
    
    def test_process_data_with_insufficient_data(self):
        """Test processing with insufficient data."""
        processor = DataProcessor(min_trading_days_for_sma=200)
        
        # Create minimal raw data
        minimal_price_data = [
            PriceData(
                date=date(2023, 1, 1),
                open=Decimal('100.00'),
                high=Decimal('105.00'),
                low=Decimal('98.00'),
                close=Decimal('102.00'),
                volume=1000000
            )
        ]
        
        raw_data = RawAPIData(
            ticker="MINIMAL",
            price_data=minimal_price_data,
            fundamental_data=None,
            info_data=None,
            data_source="yfinance",
            fetch_timestamp=pd.Timestamp.now(),
            data_quality_issues=[]
        )
        
        processed_data = processor.process_data(raw_data)
        
        assert processed_data.ticker == "MINIMAL"
        assert len(processed_data.data) == 1
        # SMA columns should be NaN due to insufficient data
        assert processed_data.data[0].get('sma_50') is None
        assert processed_data.data[0].get('sma_200') is None
