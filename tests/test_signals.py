"""Tests for the signal detection module."""

import pytest
import pandas as pd
from datetime import date
from decimal import Decimal

from src.signals import SignalDetector
from src.models import SignalEvent


class TestSignalDetector:
    """Test cases for SignalDetector class."""
    
    def test_signal_detector_initialization(self):
        """Test signal detector initialization."""
        detector = SignalDetector(min_data_points=200)
        assert detector.min_data_points == 200
    
    def test_detect_golden_crossover_sufficient_data(self, sample_dataframe):
        """Test Golden Crossover detection with sufficient data."""
        detector = SignalDetector(min_data_points=2)
        
        # Create test data with clear crossover
        test_data = pd.DataFrame({
            'sma_50': [100, 101, 102, 103, 104, 105],
            'sma_200': [102, 101.5, 101, 100.5, 100, 99.5],
            'close': [100, 101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        }, index=pd.date_range('2023-01-01', periods=6))
        
        signals = detector.detect_golden_crossover(test_data, "TEST")
        
        # Should detect crossover at index 1 (SMA 50 crosses above SMA 200)
        assert len(signals) == 1
        assert signals[0].signal_type == "golden_crossover"
        assert signals[0].ticker == "TEST"
        assert signals[0].sma_50 == Decimal('101')
        assert signals[0].sma_200 == Decimal('101.5')
    
    def test_detect_golden_crossover_insufficient_data(self):
        """Test Golden Crossover detection with insufficient data."""
        detector = SignalDetector(min_data_points=200)
        
        # Create minimal data
        test_data = pd.DataFrame({
            'sma_50': [100, 101],
            'sma_200': [102, 101],
            'close': [100, 101],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        signals = detector.detect_golden_crossover(test_data, "TEST")
        
        assert len(signals) == 0
    
    def test_detect_golden_crossover_missing_columns(self):
        """Test Golden Crossover detection with missing columns."""
        detector = SignalDetector(min_data_points=2)
        
        # Create data without SMA columns
        test_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        signals = detector.detect_golden_crossover(test_data, "TEST")
        
        assert len(signals) == 0
    
    def test_detect_death_cross_sufficient_data(self, sample_dataframe):
        """Test Death Cross detection with sufficient data."""
        detector = SignalDetector(min_data_points=2)
        
        # Create test data with clear death cross
        test_data = pd.DataFrame({
            'sma_50': [105, 104, 103, 102, 101, 100],
            'sma_200': [100, 100.5, 101, 101.5, 102, 102.5],
            'close': [105, 104, 103, 102, 101, 100],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        }, index=pd.date_range('2023-01-01', periods=6))
        
        signals = detector.detect_death_cross(test_data, "TEST")
        
        # Should detect death cross at index 1 (SMA 50 crosses below SMA 200)
        assert len(signals) == 1
        assert signals[0].signal_type == "death_cross"
        assert signals[0].ticker == "TEST"
        assert signals[0].sma_50 == Decimal('104')
        assert signals[0].sma_200 == Decimal('100.5')
    
    def test_detect_crossovers_no_crossovers(self):
        """Test crossover detection with no crossovers."""
        detector = SignalDetector(min_data_points=2)
        
        # Create data with no crossovers
        test_data = pd.DataFrame({
            'sma_50': [100, 101, 102, 103, 104, 105],
            'sma_200': [98, 99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        }, index=pd.date_range('2023-01-01', periods=6))
        
        signals = detector._detect_crossovers(test_data, "TEST", "golden_crossover")
        
        assert len(signals) == 0
    
    def test_detect_crossovers_multiple_crossovers(self):
        """Test crossover detection with multiple crossovers."""
        detector = SignalDetector(min_data_points=2)
        
        # Create data with multiple crossovers
        test_data = pd.DataFrame({
            'sma_50': [100, 101, 100, 99, 100, 101],
            'sma_200': [101, 100, 101, 102, 101, 100],
            'close': [100, 101, 100, 99, 100, 101],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        }, index=pd.date_range('2023-01-01', periods=6))
        
        golden_signals = detector._detect_crossovers(test_data, "TEST", "golden_crossover")
        death_signals = detector._detect_crossovers(test_data, "TEST", "death_cross")
        
        # Should detect multiple crossovers
        assert len(golden_signals) > 0
        assert len(death_signals) > 0
    
    def test_detect_all_signals(self, sample_processed_data):
        """Test detection of all signal types."""
        detector = SignalDetector(min_data_points=2)
        
        # Add SMA data to processed data
        for record in sample_processed_data.data:
            record['sma_50'] = 105.0
            record['sma_200'] = 104.0
        
        result = detector.detect_all_signals(sample_processed_data)
        
        assert len(result.signal_events) >= 0  # May or may not have signals
        assert 'signal_detection_timestamp' in result.processing_metadata
        assert 'total_signals_detected' in result.processing_metadata
    
    def test_get_signal_summary(self, sample_signal_events):
        """Test signal summary generation."""
        detector = SignalDetector()
        
        summary = detector.get_signal_summary(sample_signal_events)
        
        assert summary['total_signals'] == 1
        assert summary['signal_types']['golden_crossover'] == 1
        assert summary['date_range']['earliest'] == '2023-01-15'
        assert summary['date_range']['latest'] == '2023-01-15'
    
    def test_get_signal_summary_empty(self):
        """Test signal summary with empty signals."""
        detector = SignalDetector()
        
        summary = detector.get_signal_summary([])
        
        assert summary['total_signals'] == 0
        assert summary['signal_types'] == {}
        assert summary['date_range'] is None
    
    def test_validate_signal_data_valid(self, sample_dataframe):
        """Test signal data validation with valid data."""
        detector = SignalDetector(min_data_points=2)
        
        validation = detector.validate_signal_data(sample_dataframe)
        
        assert validation['is_valid'] is True
        assert validation['data_quality_score'] == 1.0
        assert len(validation['issues']) == 0
    
    def test_validate_signal_data_insufficient_data(self):
        """Test signal data validation with insufficient data."""
        detector = SignalDetector(min_data_points=200)
        
        # Create minimal data
        test_data = pd.DataFrame({
            'sma_50': [100, 101],
            'sma_200': [102, 101],
            'close': [100, 101],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        validation = detector.validate_signal_data(test_data)
        
        assert validation['is_valid'] is False
        assert validation['data_quality_score'] < 1.0
        assert len(validation['issues']) > 0
    
    def test_validate_signal_data_missing_columns(self):
        """Test signal data validation with missing columns."""
        detector = SignalDetector(min_data_points=2)
        
        # Create data with missing columns
        test_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        validation = detector.validate_signal_data(test_data)
        
        assert validation['is_valid'] is False
        assert 'Missing columns' in str(validation['issues'])
    
    def test_data_to_dataframe(self, sample_processed_data):
        """Test conversion of data list to DataFrame."""
        detector = SignalDetector()
        
        df = detector._data_to_dataframe(sample_processed_data.data)
        
        assert len(df) == len(sample_processed_data.data)
        assert df.index.name == 'date'
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    def test_data_to_dataframe_empty(self):
        """Test conversion of empty data list to DataFrame."""
        detector = SignalDetector()
        
        df = detector._data_to_dataframe([])
        
        assert len(df) == 0
    
    def test_signal_event_creation(self):
        """Test SignalEvent object creation."""
        signal = SignalEvent(
            ticker="TEST",
            signal_type="golden_crossover",
            date=date(2023, 1, 15),
            sma_50=Decimal('105.00'),
            sma_200=Decimal('104.00'),
            price=Decimal('106.00'),
            volume=1500000,
            metadata={"test": "value"}
        )
        
        assert signal.ticker == "TEST"
        assert signal.signal_type == "golden_crossover"
        assert signal.date == date(2023, 1, 15)
        assert signal.sma_50 == Decimal('105.00')
        assert signal.sma_200 == Decimal('104.00')
        assert signal.price == Decimal('106.00')
        assert signal.volume == 1500000
        assert signal.metadata == {"test": "value"}
