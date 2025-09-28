"""Signal detection module for trading signals like Golden Crossover and Death Cross."""

import logging
from datetime import date
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from decimal import Decimal

from .models import SignalEvent, ProcessedDataFrame


logger = logging.getLogger(__name__)


class SignalDetector:
    """Detects trading signals from processed financial data."""
    
    def __init__(self, min_data_points: int = 200):
        """Initialize the signal detector.
        
        Args:
            min_data_points: Minimum data points required for signal detection
        """
        self.min_data_points = min_data_points
    
    def detect_golden_crossover(self, df: pd.DataFrame, ticker: str) -> List[SignalEvent]:
        """Detect Golden Crossover signals (50-day SMA crosses above 200-day SMA).
        
        Args:
            df: DataFrame with price data and SMA indicators
            ticker: Stock ticker symbol
            
        Returns:
            List of SignalEvent objects for Golden Crossovers
        """
        logger.info(f"Detecting Golden Crossover signals for {ticker}")
        
        if len(df) < self.min_data_points:
            logger.warning(f"Insufficient data for signal detection. "
                          f"Available: {len(df)} days, Required: {self.min_data_points} days")
            return []
        
        # Check if required columns exist
        if 'sma_50' not in df.columns or 'sma_200' not in df.columns:
            logger.error("Missing SMA columns for signal detection")
            return []
        
        # Remove rows with NaN values in SMA columns
        sma_data = df[['sma_50', 'sma_200', 'close', 'volume']].dropna()
        
        if len(sma_data) < 2:
            logger.warning("Insufficient valid SMA data for signal detection")
            return []
        
        # Detect crossovers using vectorized operations
        crossover_signals = self._detect_crossovers(sma_data, ticker, "golden_crossover")
        
        logger.info(f"Found {len(crossover_signals)} Golden Crossover signals for {ticker}")
        return crossover_signals
    
    def detect_death_cross(self, df: pd.DataFrame, ticker: str) -> List[SignalEvent]:
        """Detect Death Cross signals (50-day SMA crosses below 200-day SMA).
        
        Args:
            df: DataFrame with price data and SMA indicators
            ticker: Stock ticker symbol
            
        Returns:
            List of SignalEvent objects for Death Crosses
        """
        logger.info(f"Detecting Death Cross signals for {ticker}")
        
        if len(df) < self.min_data_points:
            logger.warning(f"Insufficient data for signal detection. "
                          f"Available: {len(df)} days, Required: {self.min_data_points} days")
            return []
        
        # Check if required columns exist
        if 'sma_50' not in df.columns or 'sma_200' not in df.columns:
            logger.error("Missing SMA columns for signal detection")
            return []
        
        # Remove rows with NaN values in SMA columns
        sma_data = df[['sma_50', 'sma_200', 'close', 'volume']].dropna()
        
        if len(sma_data) < 2:
            logger.warning("Insufficient valid SMA data for signal detection")
            return []
        
        # Detect crossovers using vectorized operations
        death_cross_signals = self._detect_crossovers(sma_data, ticker, "death_cross")
        
        logger.info(f"Found {len(death_cross_signals)} Death Cross signals for {ticker}")
        return death_cross_signals
    
    def _detect_crossovers(self, sma_data: pd.DataFrame, ticker: str, 
                          signal_type: str) -> List[SignalEvent]:
        """Detect crossover signals using vectorized operations.
        
        Args:
            sma_data: DataFrame with SMA data
            ticker: Stock ticker symbol
            signal_type: Type of signal to detect ('golden_crossover' or 'death_cross')
            
        Returns:
            List of SignalEvent objects
        """
        signals = []
        
        # Calculate the difference between SMAs
        sma_diff = sma_data['sma_50'] - sma_data['sma_200']
        
        # Detect crossover points
        if signal_type == "golden_crossover":
            # Golden Crossover: SMA 50 crosses above SMA 200
            crossover_mask = (sma_diff > 0) & (sma_diff.shift(1) <= 0)
        elif signal_type == "death_cross":
            # Death Cross: SMA 50 crosses below SMA 200
            crossover_mask = (sma_diff < 0) & (sma_diff.shift(1) >= 0)
        else:
            logger.error(f"Unknown signal type: {signal_type}")
            return []
        
        # Get crossover dates
        crossover_dates = sma_data[crossover_mask].index
        
        # Create signal events
        for crossover_date in crossover_dates:
            try:
                row = sma_data.loc[crossover_date]
                
                signal_event = SignalEvent(
                    ticker=ticker,
                    signal_type=signal_type,
                    date=crossover_date.date(),
                    sma_50=Decimal(str(row['sma_50'])),
                    sma_200=Decimal(str(row['sma_200'])),
                    price=Decimal(str(row['close'])),
                    volume=int(row['volume']),
                    metadata={
                        "detection_method": "vectorized_crossover",
                        "sma_difference": float(sma_diff.loc[crossover_date]),
                        "previous_sma_difference": float(sma_diff.shift(1).loc[crossover_date]) if crossover_date != sma_data.index[0] else None
                    }
                )
                signals.append(signal_event)
                
            except Exception as e:
                logger.warning(f"Error creating signal event for {crossover_date}: {str(e)}")
                continue
        
        return signals
    
    def detect_all_signals(self, processed_data: ProcessedDataFrame) -> ProcessedDataFrame:
        """Detect all types of signals for the processed data.
        
        Args:
            processed_data: ProcessedDataFrame with calculated metrics
            
        Returns:
            ProcessedDataFrame with signal events added
        """
        logger.info(f"Detecting all signals for {processed_data.ticker}")
        
        # Convert data back to DataFrame for signal detection
        df = self._data_to_dataframe(processed_data.data)
        
        all_signals = []
        
        # Detect Golden Crossover
        golden_crossovers = self.detect_golden_crossover(df, processed_data.ticker)
        all_signals.extend(golden_crossovers)
        
        # Detect Death Cross
        death_crosses = self.detect_death_cross(df, processed_data.ticker)
        all_signals.extend(death_crosses)
        
        # Sort signals by date
        all_signals.sort(key=lambda x: x.date)
        
        # Update processed data with signals
        processed_data.signal_events = all_signals
        
        # Add signal detection metadata
        processed_data.processing_metadata.update({
            "signal_detection_timestamp": pd.Timestamp.now().isoformat(),
            "total_signals_detected": len(all_signals),
            "golden_crossover_count": len(golden_crossovers),
            "death_cross_count": len(death_crosses)
        })
        
        logger.info(f"Signal detection completed for {processed_data.ticker}: "
                   f"{len(all_signals)} total signals detected")
        
        return processed_data
    
    def _data_to_dataframe(self, data_list: List[Dict]) -> pd.DataFrame:
        """Convert data list back to DataFrame for signal detection.
        
        Args:
            data_list: List of data dictionaries
            
        Returns:
            DataFrame with date index
        """
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def get_signal_summary(self, signals: List[SignalEvent]) -> Dict[str, Any]:
        """Get summary of detected signals.
        
        Args:
            signals: List of SignalEvent objects
            
        Returns:
            Dictionary with signal summary
        """
        if not signals:
            return {
                "total_signals": 0,
                "signal_types": {},
                "date_range": None
            }
        
        # Count signals by type
        signal_types = {}
        for signal in signals:
            signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1
        
        # Get date range
        dates = [signal.date for signal in signals]
        date_range = {
            "earliest": min(dates).isoformat(),
            "latest": max(dates).isoformat()
        }
        
        return {
            "total_signals": len(signals),
            "signal_types": signal_types,
            "date_range": date_range
        }
    
    def validate_signal_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality for signal detection.
        
        Args:
            df: DataFrame with price and indicator data
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "data_quality_score": 0.0
        }
        
        # Check data length
        if len(df) < self.min_data_points:
            validation_results["is_valid"] = False
            validation_results["issues"].append(
                f"Insufficient data: {len(df)} days available, {self.min_data_points} required"
            )
        
        # Check for required columns
        required_columns = ['sma_50', 'sma_200', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Missing columns: {missing_columns}")
        
        # Check for NaN values in critical columns
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            sma_data = df[['sma_50', 'sma_200']].dropna()
            nan_percentage = (len(df) - len(sma_data)) / len(df) * 100
            if nan_percentage > 50:
                validation_results["issues"].append(
                    f"High percentage of NaN values in SMA data: {nan_percentage:.1f}%"
                )
        
        # Calculate data quality score
        if validation_results["is_valid"]:
            validation_results["data_quality_score"] = 1.0
        else:
            validation_results["data_quality_score"] = max(0.0, 1.0 - len(validation_results["issues"]) * 0.2)
        
        return validation_results
