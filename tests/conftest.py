"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
from datetime import date, datetime
from decimal import Decimal
from typing import List, Dict, Any

from src.models import PriceData, FundamentalData, RawAPIData, ProcessedDataFrame, SignalEvent


@pytest.fixture
def sample_price_data() -> List[PriceData]:
    """Sample price data for testing."""
    return [
        PriceData(
            date=date(2023, 1, 1),
            open=Decimal('100.00'),
            high=Decimal('105.00'),
            low=Decimal('98.00'),
            close=Decimal('102.00'),
            volume=1000000
        ),
        PriceData(
            date=date(2023, 1, 2),
            open=Decimal('102.00'),
            high=Decimal('108.00'),
            low=Decimal('101.00'),
            close=Decimal('106.00'),
            volume=1200000
        ),
        PriceData(
            date=date(2023, 1, 3),
            open=Decimal('106.00'),
            high=Decimal('110.00'),
            low=Decimal('104.00'),
            close=Decimal('108.00'),
            volume=1100000
        )
    ]


@pytest.fixture
def sample_fundamental_data() -> List[FundamentalData]:
    """Sample fundamental data for testing."""
    return [
        FundamentalData(
            date=date(2023, 1, 1),
            total_assets=Decimal('1000000000'),
            total_liabilities=Decimal('400000000'),
            shareholders_equity=Decimal('600000000'),
            book_value_per_share=Decimal('10.00'),
            market_cap=Decimal('5000000000'),
            enterprise_value=Decimal('4500000000')
        )
    ]


@pytest.fixture
def sample_raw_data(sample_price_data, sample_fundamental_data) -> RawAPIData:
    """Sample raw API data for testing."""
    return RawAPIData(
        ticker="TEST",
        price_data=sample_price_data,
        fundamental_data=sample_fundamental_data,
        info_data={"marketCap": 5000000000, "bookValue": 10.00},
        data_source="yfinance",
        fetch_timestamp=datetime.now(),
        data_quality_issues=[]
    )


@pytest.fixture
def sample_processed_data() -> ProcessedDataFrame:
    """Sample processed data for testing."""
    data = [
        {
            'date': '2023-01-01',
            'open': 100.0,
            'high': 105.0,
            'low': 98.0,
            'close': 102.0,
            'volume': 1000000,
            'sma_50': 101.0,
            'sma_200': 100.5,
            'high_52w': 110.0,
            'pct_from_52w_high': -7.27
        },
        {
            'date': '2023-01-02',
            'open': 102.0,
            'high': 108.0,
            'low': 101.0,
            'close': 106.0,
            'volume': 1200000,
            'sma_50': 102.0,
            'sma_200': 100.8,
            'high_52w': 110.0,
            'pct_from_52w_high': -3.64
        }
    ]
    
    return ProcessedDataFrame(
        ticker="TEST",
        data=data,
        signal_events=[],
        data_quality_notes=[],
        processing_metadata={}
    )


@pytest.fixture
def sample_signal_events() -> List[SignalEvent]:
    """Sample signal events for testing."""
    return [
        SignalEvent(
            ticker="TEST",
            signal_type="golden_crossover",
            date=date(2023, 1, 15),
            sma_50=Decimal('105.00'),
            sma_200=Decimal('104.00'),
            price=Decimal('106.00'),
            volume=1500000,
            metadata={"detection_method": "vectorized_crossover"}
        )
    ]


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    data = {
        'date': pd.date_range('2023-01-01', periods=5, freq='D'),
        'open': [100, 102, 104, 106, 108],
        'high': [105, 107, 109, 111, 113],
        'low': [98, 100, 102, 104, 106],
        'close': [102, 104, 106, 108, 110],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
        'sma_50': [101, 102, 103, 104, 105],
        'sma_200': [100, 100.5, 101, 101.5, 102]
    }
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance data for testing."""
    return {
        'history': pd.DataFrame({
            'Open': [100, 102, 104],
            'High': [105, 107, 109],
            'Low': [98, 100, 102],
            'Close': [102, 104, 106],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3)),
        'info': {
            'marketCap': 5000000000,
            'bookValue': 10.0,
            'enterpriseValue': 4500000000
        },
        'quarterly_balance_sheet': pd.DataFrame({
            'Total Assets': [1000000000, 1100000000],
            'Total Liab': [400000000, 450000000],
            'Stockholders Equity': [600000000, 650000000]
        }, index=pd.date_range('2023-01-01', periods=2))
    }
