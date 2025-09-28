"""Pydantic models for data validation and type safety."""

from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import pandas as pd


class PriceData(BaseModel):
    """Model for daily OHLCV price data."""
    
    date: date
    open: Decimal = Field(gt=0, description="Opening price must be positive")
    high: Decimal = Field(gt=0, description="High price must be positive")
    low: Decimal = Field(gt=0, description="Low price must be positive")
    close: Decimal = Field(gt=0, description="Closing price must be positive")
    volume: int = Field(ge=0, description="Volume must be non-negative")
    
    @validator('high')
    def high_must_be_highest(cls, v, values):
        """Validate that high is the highest price."""
        if 'open' in values and v < values['open']:
            raise ValueError('High must be >= Open')
        if 'low' in values and v < values['low']:
            raise ValueError('High must be >= Low')
        if 'close' in values and v < values['close']:
            raise ValueError('High must be >= Close')
        return v
    
    @validator('low')
    def low_must_be_lowest(cls, v, values):
        """Validate that low is the lowest price."""
        if 'open' in values and v > values['open']:
            raise ValueError('Low must be <= Open')
        if 'close' in values and v > values['close']:
            raise ValueError('Low must be <= Close')
        return v


class FundamentalData(BaseModel):
    """Model for fundamental financial data."""
    
    date: date
    total_assets: Optional[Decimal] = None
    total_liabilities: Optional[Decimal] = None
    shareholders_equity: Optional[Decimal] = None
    book_value_per_share: Optional[Decimal] = None
    market_cap: Optional[Decimal] = None
    enterprise_value: Optional[Decimal] = None
    shares_outstanding: Optional[int] = None
    
    @validator('book_value_per_share')
    def bvps_must_be_positive(cls, v):
        """Validate book value per share is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError('Book value per share must be positive')
        return v


class TechnicalIndicators(BaseModel):
    """Model for calculated technical indicators."""
    
    date: date
    sma_50: Optional[Decimal] = None
    sma_200: Optional[Decimal] = None
    high_52w: Optional[Decimal] = None
    pct_from_52w_high: Optional[Decimal] = None
    price_to_book: Optional[Decimal] = None
    book_value_per_share: Optional[Decimal] = None
    enterprise_value: Optional[Decimal] = None


class SignalEvent(BaseModel):
    """Model for trading signal events."""
    
    ticker: str
    signal_type: str = Field(..., pattern="^(golden_crossover|death_cross)$")
    date: date
    sma_50: Decimal
    sma_200: Decimal
    price: Decimal
    volume: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedDataFrame(BaseModel):
    """Model for the final processed DataFrame."""
    
    ticker: str
    data: List[Dict[str, Any]]
    signal_events: List[SignalEvent]
    data_quality_notes: List[str] = Field(default_factory=list)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('data')
    def validate_dataframe_structure(cls, v):
        """Validate that data contains required columns."""
        if not v:
            raise ValueError('Data cannot be empty')
        
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for row in v:
            for col in required_columns:
                if col not in row:
                    raise ValueError(f'Missing required column: {col}')
        return v


class RawAPIData(BaseModel):
    """Model for raw API response data."""
    
    ticker: str
    price_data: List[PriceData]
    fundamental_data: Optional[List[FundamentalData]] = None
    info_data: Optional[Dict[str, Any]] = None
    data_source: str = "yfinance"
    fetch_timestamp: datetime
    data_quality_issues: List[str] = Field(default_factory=list)


class DatabaseTicker(BaseModel):
    """Model for ticker information in database."""
    
    ticker: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    country: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    first_listed: Optional[date] = None
    last_updated: datetime


class DatabaseDailyMetrics(BaseModel):
    """Model for daily metrics stored in database."""
    
    ticker: str
    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    sma_50: Optional[Decimal] = None
    sma_200: Optional[Decimal] = None
    high_52w: Optional[Decimal] = None
    pct_from_52w_high: Optional[Decimal] = None
    price_to_book: Optional[Decimal] = None
    book_value_per_share: Optional[Decimal] = None
    enterprise_value: Optional[Decimal] = None


class DatabaseSignalEvent(BaseModel):
    """Model for signal events stored in database."""
    
    ticker: str
    signal_type: str
    date: date
    sma_50: Decimal
    sma_200: Decimal
    price: Decimal
    volume: int
    signal_metadata: str  # JSON string
    created_at: datetime


class ExportData(BaseModel):
    """Model for final JSON export."""
    
    ticker: str
    analysis_date: datetime
    data_period: Dict[str, Any]
    technical_indicators: List[TechnicalIndicators]
    signal_events: List[SignalEvent]
    fundamental_metrics: Optional[Dict[str, Any]] = None
    data_quality_notes: List[str] = Field(default_factory=list)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
