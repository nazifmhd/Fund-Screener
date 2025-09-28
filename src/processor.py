"""Data processing module for calculating technical indicators and fundamental ratios."""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from decimal import Decimal

from .models import RawAPIData, TechnicalIndicators, ProcessedDataFrame, PriceData, FundamentalData


logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data processing, merging, and calculation of technical indicators."""
    
    def __init__(self, min_trading_days_for_sma: int = 200, forward_fill_fundamentals: bool = True):
        """Initialize the data processor.
        
        Args:
            min_trading_days_for_sma: Minimum trading days required for SMA calculation
            forward_fill_fundamentals: Whether to forward-fill fundamental data
        """
        self.min_trading_days_for_sma = min_trading_days_for_sma
        self.forward_fill_fundamentals = forward_fill_fundamentals
    
    def process_data(self, raw_data: RawAPIData) -> ProcessedDataFrame:
        """Process raw API data into calculated metrics and indicators.
        
        Args:
            raw_data: Raw data from API
            
        Returns:
            ProcessedDataFrame with calculated metrics
        """
        logger.info(f"Processing data for {raw_data.ticker}")
        
        # Convert price data to DataFrame
        price_df = self._create_price_dataframe(raw_data.price_data)
        
        # Process fundamental data
        fundamental_df = self._process_fundamental_data(raw_data.fundamental_data, raw_data.info_data)
        
        # Merge price and fundamental data
        merged_df = self._merge_price_and_fundamental_data(price_df, fundamental_df)
        
        # Calculate technical indicators
        processed_df = self._calculate_technical_indicators(merged_df)
        
        # Calculate fundamental ratios
        processed_df = self._calculate_fundamental_ratios(processed_df, raw_data.info_data)
        
        # Convert to list of dictionaries for JSON serialization
        data_list = self._dataframe_to_dict_list(processed_df)
        
        # Create processing metadata
        processing_metadata = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_records": len(data_list),
            "price_records": len(raw_data.price_data),
            "fundamental_records": len(raw_data.fundamental_data) if raw_data.fundamental_data else 0,
            "data_quality_issues": raw_data.data_quality_issues,
            "forward_fill_used": self.forward_fill_fundamentals
        }
        
        return ProcessedDataFrame(
            ticker=raw_data.ticker,
            data=data_list,
            signal_events=[],  # Will be populated by signal detector
            data_quality_notes=raw_data.data_quality_issues,
            processing_metadata=processing_metadata
        )
    
    def _create_price_dataframe(self, price_data: List[PriceData]) -> pd.DataFrame:
        """Convert price data to DataFrame.
        
        Args:
            price_data: List of PriceData objects
            
        Returns:
            DataFrame with price data
        """
        data = []
        for price in price_data:
            data.append({
                'date': price.date,
                'open': float(price.open),
                'high': float(price.high),
                'low': float(price.low),
                'close': float(price.close),
                'volume': price.volume
            })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _process_fundamental_data(self, fundamental_data: Optional[List[FundamentalData]], 
                                 info_data: Optional[Dict]) -> Optional[pd.DataFrame]:
        """Process fundamental data into DataFrame.
        
        Args:
            fundamental_data: List of FundamentalData objects
            info_data: Additional info data
            
        Returns:
            DataFrame with fundamental data or None
        """
        if not fundamental_data:
            logger.warning("No fundamental data available")
            return None
        
        data = []
        for fund in fundamental_data:
            row = {
                'date': fund.date,
                'total_assets': float(fund.total_assets) if fund.total_assets else None,
                'total_liabilities': float(fund.total_liabilities) if fund.total_liabilities else None,
                'shareholders_equity': float(fund.shareholders_equity) if fund.shareholders_equity else None,
                'book_value_per_share': float(fund.book_value_per_share) if fund.book_value_per_share else None,
                'market_cap': float(fund.market_cap) if fund.market_cap else None,
                'enterprise_value': float(fund.enterprise_value) if fund.enterprise_value else None,
                'shares_outstanding': fund.shares_outstanding
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Forward-fill fundamental data if enabled
        if self.forward_fill_fundamentals:
            df = df.ffill()
            logger.info("Applied forward-fill to fundamental data")
        
        return df
    
    def _merge_price_and_fundamental_data(self, price_df: pd.DataFrame, 
                                        fundamental_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Merge price and fundamental data.
        
        Args:
            price_df: Price data DataFrame
            fundamental_df: Fundamental data DataFrame
            
        Returns:
            Merged DataFrame
        """
        if fundamental_df is not None:
            # Merge on date index
            merged_df = price_df.join(fundamental_df, how='left')
            logger.info(f"Merged price and fundamental data. "
                       f"Fundamental data available for {fundamental_df.index.nunique()} dates")
        else:
            merged_df = price_df.copy()
            logger.info("No fundamental data to merge")
        
        return merged_df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with technical indicators added
        """
        result_df = df.copy()
        
        # Calculate moving averages
        result_df['sma_50'] = self._calculate_sma(df['close'], 50)
        result_df['sma_200'] = self._calculate_sma(df['close'], 200)
        
        # Calculate 52-week high and percentage from high
        result_df['high_52w'] = df['high'].rolling(window=252, min_periods=1).max()
        result_df['pct_from_52w_high'] = ((df['close'] - result_df['high_52w']) / result_df['high_52w'] * 100)
        
        # Log calculation results
        sma_50_count = result_df['sma_50'].notna().sum()
        sma_200_count = result_df['sma_200'].notna().sum()
        
        logger.info(f"Calculated technical indicators: "
                   f"SMA 50 ({sma_50_count} values), "
                   f"SMA 200 ({sma_200_count} values)")
        
        return result_df
    
    def _calculate_sma(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average.
        
        Args:
            series: Price series
            window: Window size for SMA
            
        Returns:
            SMA series
        """
        if len(series) < window:
            logger.warning(f"Insufficient data for {window}-day SMA. "
                          f"Available: {len(series)} days, Required: {window} days")
            return pd.Series(index=series.index, dtype=float)
        
        return series.rolling(window=window, min_periods=window).mean()
    
    def _calculate_fundamental_ratios(self, df: pd.DataFrame, 
                                    info_data: Optional[Dict]) -> pd.DataFrame:
        """Calculate fundamental ratios.
        
        Args:
            df: DataFrame with price and fundamental data
            info_data: Additional info data
            
        Returns:
            DataFrame with fundamental ratios added
        """
        result_df = df.copy()
        
        # Calculate Price-to-Book ratio
        if 'book_value_per_share' in df.columns:
            result_df['price_to_book'] = df['close'] / df['book_value_per_share']
        
        # Calculate Enterprise Value if market cap is available
        if 'market_cap' in df.columns and 'total_liabilities' in df.columns:
            # Simplified EV calculation: Market Cap + Total Liabilities
            result_df['enterprise_value'] = df['market_cap'] + df['total_liabilities']
        elif info_data and 'enterpriseValue' in info_data:
            # Use enterprise value from info data
            ev_value = info_data.get('enterpriseValue')
            if ev_value and not pd.isna(ev_value):
                result_df['enterprise_value'] = float(ev_value)
        
        # Add book value per share if not already present
        if 'book_value_per_share' not in df.columns and 'shareholders_equity' in df.columns and 'shares_outstanding' in df.columns:
            result_df['book_value_per_share'] = df['shareholders_equity'] / df['shares_outstanding']
        
        # Log calculation results
        pbr_count = result_df['price_to_book'].notna().sum() if 'price_to_book' in result_df.columns else 0
        ev_count = result_df['enterprise_value'].notna().sum() if 'enterprise_value' in result_df.columns else 0
        
        logger.info(f"Calculated fundamental ratios: "
                   f"P/B ratio ({pbr_count} values), "
                   f"Enterprise Value ({ev_count} values)")
        
        return result_df
    
    def _dataframe_to_dict_list(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to list of dictionaries for JSON serialization.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            List of dictionaries
        """
        # Reset index to include date as a column
        df_reset = df.reset_index()
        
        # Convert to list of dictionaries
        data_list = []
        for _, row in df_reset.iterrows():
            record = {}
            for col, value in row.items():
                if pd.isna(value):
                    record[col] = None
                elif isinstance(value, (np.integer, np.floating)):
                    record[col] = float(value)
                elif isinstance(value, np.datetime64):
                    record[col] = pd.to_datetime(value).date().isoformat()
                else:
                    record[col] = value
            data_list.append(record)
        
        return data_list
    
    def get_data_quality_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get data quality summary for the processed data.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        summary = {
            "total_records": len(df),
            "date_range": {
                "start": df.index.min().isoformat() if not df.empty else None,
                "end": df.index.max().isoformat() if not df.empty else None
            },
            "missing_data": {},
            "data_completeness": {}
        }
        
        # Check for missing data in key columns
        key_columns = ['close', 'volume', 'sma_50', 'sma_200']
        for col in key_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                summary["missing_data"][col] = missing_count
                summary["data_completeness"][col] = (len(df) - missing_count) / len(df) * 100
        
        return summary
