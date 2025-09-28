"""Data fetcher module for retrieving stock data from yfinance API."""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import yfinance as yf
import pandas as pd
from decimal import Decimal
import time

from .models import RawAPIData, PriceData, FundamentalData


logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles data fetching from yfinance API with error handling and validation."""
    
    def __init__(self, timeout: int = 30, retry_attempts: int = 3, retry_delay: int = 1):
        """Initialize the data fetcher.
        
        Args:
            timeout: API timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
        """
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
    
    def fetch_stock_data(self, ticker: str, period: str = "5y") -> RawAPIData:
        """Fetch comprehensive stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'NVDA', 'RELIANCE.NS')
            period: Historical period to fetch ('5y', '1y', etc.)
            
        Returns:
            RawAPIData object containing validated price and fundamental data
            
        Raises:
            ValueError: If ticker is invalid or data cannot be fetched
        """
        logger.info(f"Fetching data for ticker: {ticker}")
        
        # Validate ticker format
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        data_quality_issues = []
        
        try:
            # Fetch ticker object with retry logic
            ticker_obj = self._fetch_ticker_with_retry(ticker)
            
            # Fetch price data
            price_data = self._fetch_price_data(ticker_obj, period)
            if not price_data:
                raise ValueError(f"No price data available for {ticker}")
            
            # Fetch fundamental data with fallback strategy
            fundamental_data = self._fetch_fundamental_data(ticker_obj, data_quality_issues)
            
            # Fetch info data
            info_data = self._fetch_info_data(ticker_obj, data_quality_issues)
            
            # Create and validate the response
            raw_data = RawAPIData(
                ticker=ticker,
                price_data=price_data,
                fundamental_data=fundamental_data,
                info_data=info_data,
                data_source="yfinance",
                fetch_timestamp=datetime.now(),
                data_quality_issues=data_quality_issues
            )
            
            logger.info(f"Successfully fetched data for {ticker}. "
                       f"Price records: {len(price_data)}, "
                       f"Fundamental records: {len(fundamental_data) if fundamental_data else 0}")
            
            return raw_data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")
    
    def _fetch_ticker_with_retry(self, ticker: str) -> yf.Ticker:
        """Fetch ticker object with retry logic.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            yfinance Ticker object
            
        Raises:
            ValueError: If ticker cannot be fetched after retries
        """
        for attempt in range(self.retry_attempts):
            try:
                ticker_obj = yf.Ticker(ticker)
                # Test if ticker is valid by trying to get basic info
                info = ticker_obj.info
                if not info or len(info) < 5:  # Basic validation
                    raise ValueError("Invalid ticker or insufficient data")
                return ticker_obj
                
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise ValueError(f"Failed to fetch ticker {ticker} after {self.retry_attempts} attempts: {str(e)}")
                
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}. Retrying...")
                time.sleep(self.retry_delay)
        
        raise ValueError(f"Failed to fetch ticker {ticker}")
    
    def _fetch_price_data(self, ticker_obj: yf.Ticker, period: str) -> List[PriceData]:
        """Fetch and validate price data.
        
        Args:
            ticker_obj: yfinance Ticker object
            period: Historical period
            
        Returns:
            List of validated PriceData objects
        """
        try:
            # Fetch historical data
            hist = ticker_obj.history(period=period, auto_adjust=True, back_adjust=True)
            
            if hist.empty:
                logger.warning(f"No historical data available for {ticker_obj.ticker}")
                return []
            
            # Convert to our format
            price_data = []
            for date_idx, row in hist.iterrows():
                try:
                    price_record = PriceData(
                        date=date_idx.date(),
                        open=Decimal(str(row['Open'])),
                        high=Decimal(str(row['High'])),
                        low=Decimal(str(row['Low'])),
                        close=Decimal(str(row['Close'])),
                        volume=int(row['Volume'])
                    )
                    price_data.append(price_record)
                except Exception as e:
                    logger.warning(f"Skipping invalid price data for {date_idx.date()}: {str(e)}")
                    continue
            
            logger.info(f"Fetched {len(price_data)} price records")
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            return []
    
    def _fetch_fundamental_data(self, ticker_obj: yf.Ticker, 
                               data_quality_issues: List[str]) -> Optional[List[FundamentalData]]:
        """Fetch fundamental data with fallback strategy.
        
        Args:
            ticker_obj: yfinance Ticker object
            data_quality_issues: List to append quality issues to
            
        Returns:
            List of FundamentalData objects or None if no data available
        """
        fundamental_data = []
        
        try:
            # Strategy 1: Try quarterly balance sheet
            quarterly_bs = ticker_obj.quarterly_balance_sheet
            if not quarterly_bs.empty:
                fundamental_data.extend(self._process_balance_sheet(quarterly_bs, "quarterly"))
                logger.info("Using quarterly balance sheet data")
            else:
                data_quality_issues.append("No quarterly balance sheet data available")
                
                # Strategy 2: Fall back to annual balance sheet
                annual_bs = ticker_obj.balance_sheet
                if not annual_bs.empty:
                    fundamental_data.extend(self._process_balance_sheet(annual_bs, "annual"))
                    logger.info("Using annual balance sheet data (fallback)")
                    data_quality_issues.append("Using annual data as quarterly fallback")
                else:
                    data_quality_issues.append("No balance sheet data available")
            
            # Strategy 3: Use info data for basic metrics
            info_data = ticker_obj.info
            if info_data and len(info_data) > 5:
                synthetic_data = self._create_synthetic_fundamental_data(info_data)
                if synthetic_data:
                    fundamental_data.extend(synthetic_data)
                    logger.info("Using synthetic fundamental data from info")
                    data_quality_issues.append("Using synthetic fundamental data")
            
            return fundamental_data if fundamental_data else None
            
        except Exception as e:
            logger.warning(f"Error fetching fundamental data: {str(e)}")
            data_quality_issues.append(f"Error fetching fundamental data: {str(e)}")
            return None
    
    def _process_balance_sheet(self, balance_sheet: pd.DataFrame, 
                              data_type: str) -> List[FundamentalData]:
        """Process balance sheet data into FundamentalData objects.
        
        Args:
            balance_sheet: Balance sheet DataFrame
            data_type: Type of data ('quarterly' or 'annual')
            
        Returns:
            List of FundamentalData objects
        """
        fundamental_data = []
        
        for date_idx, row in balance_sheet.iterrows():
            try:
                # Extract relevant metrics
                total_assets = self._safe_decimal(row.get('Total Assets'))
                total_liabilities = self._safe_decimal(row.get('Total Liab'))
                shareholders_equity = self._safe_decimal(row.get('Stockholders Equity'))
                
                # Calculate book value per share if possible
                book_value_per_share = None
                if shareholders_equity and 'Shares Outstanding' in row:
                    shares = self._safe_decimal(row.get('Shares Outstanding'))
                    if shares and shares > 0:
                        book_value_per_share = shareholders_equity / shares
                
                fundamental_record = FundamentalData(
                    date=date_idx.date(),
                    total_assets=total_assets,
                    total_liabilities=total_liabilities,
                    shareholders_equity=shareholders_equity,
                    book_value_per_share=book_value_per_share
                )
                fundamental_data.append(fundamental_record)
                
            except Exception as e:
                logger.warning(f"Skipping invalid fundamental data for {date_idx.date()}: {str(e)}")
                continue
        
        return fundamental_data
    
    def _create_synthetic_fundamental_data(self, info_data: Dict[str, Any]) -> List[FundamentalData]:
        """Create synthetic fundamental data from info data.
        
        Args:
            info_data: Info dictionary from yfinance
            
        Returns:
            List of FundamentalData objects
        """
        try:
            # Extract basic metrics
            market_cap = self._safe_decimal(info_data.get('marketCap'))
            enterprise_value = self._safe_decimal(info_data.get('enterpriseValue'))
            book_value = self._safe_decimal(info_data.get('bookValue'))
            shares_outstanding = self._safe_decimal(info_data.get('sharesOutstanding'))
            
            # Calculate book value per share
            book_value_per_share = None
            if book_value and shares_outstanding and shares_outstanding > 0:
                book_value_per_share = book_value / shares_outstanding
            
            # Create synthetic record (using current date)
            synthetic_data = FundamentalData(
                date=date.today(),
                market_cap=market_cap,
                enterprise_value=enterprise_value,
                book_value_per_share=book_value_per_share,
                shares_outstanding=int(shares_outstanding) if shares_outstanding else None
            )
            
            return [synthetic_data]
            
        except Exception as e:
            logger.warning(f"Error creating synthetic fundamental data: {str(e)}")
            return []
    
    def _fetch_info_data(self, ticker_obj: yf.Ticker, 
                        data_quality_issues: List[str]) -> Optional[Dict[str, Any]]:
        """Fetch additional info data.
        
        Args:
            ticker_obj: yfinance Ticker object
            data_quality_issues: List to append quality issues to
            
        Returns:
            Info dictionary or None if unavailable
        """
        try:
            info = ticker_obj.info
            if info and len(info) > 5:
                return info
            else:
                data_quality_issues.append("Limited info data available")
                return None
        except Exception as e:
            logger.warning(f"Error fetching info data: {str(e)}")
            data_quality_issues.append(f"Error fetching info data: {str(e)}")
            return None
    
    def _safe_decimal(self, value: Any) -> Optional[Decimal]:
        """Safely convert value to Decimal.
        
        Args:
            value: Value to convert
            
        Returns:
            Decimal value or None if conversion fails
        """
        if value is None or pd.isna(value):
            return None
        
        try:
            if isinstance(value, (int, float)):
                return Decimal(str(value))
            elif isinstance(value, str):
                return Decimal(value)
            else:
                return None
        except (ValueError, TypeError, ArithmeticError):
            return None
