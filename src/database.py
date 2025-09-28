"""Database module for SQLite operations with SQLAlchemy ORM."""

import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from sqlalchemy import create_engine, Column, String, Date, Integer, Float, DateTime, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

from .models import (
    DatabaseTicker, DatabaseDailyMetrics, DatabaseSignalEvent,
    ProcessedDataFrame, SignalEvent, TechnicalIndicators
)


logger = logging.getLogger(__name__)

Base = declarative_base()


class Ticker(Base):
    """SQLAlchemy model for ticker information."""
    __tablename__ = 'tickers'
    
    ticker = Column(String(20), primary_key=True)
    name = Column(String(200))
    exchange = Column(String(50))
    country = Column(String(50))
    sector = Column(String(100))
    industry = Column(String(100))
    first_listed = Column(Date)
    last_updated = Column(DateTime, default=datetime.utcnow)


class DailyMetrics(Base):
    """SQLAlchemy model for daily metrics."""
    __tablename__ = 'daily_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    high_52w = Column(Float)
    pct_from_52w_high = Column(Float)
    price_to_book = Column(Float)
    book_value_per_share = Column(Float)
    enterprise_value = Column(Float)
    
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='unique_ticker_date'),
    )


class SignalEvents(Base):
    """SQLAlchemy model for signal events."""
    __tablename__ = 'signal_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    signal_type = Column(String(50), nullable=False)
    date = Column(Date, nullable=False)
    sma_50 = Column(Float, nullable=False)
    sma_200 = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Manages database operations for the financial analysis pipeline."""
    
    def __init__(self, database_path: str):
        """Initialize the database manager.
        
        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = database_path
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            # Create database directory if it doesn't exist
            db_path = Path(self.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create engine
            self.engine = create_engine(
                f"sqlite:///{self.database_path}",
                echo=False,
                pool_pre_ping=True
            )
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            logger.info(f"Database initialized at {self.database_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def get_session(self) -> Session:
        """Get a database session.
        
        Returns:
            SQLAlchemy session
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    def save_ticker_info(self, ticker_info: DatabaseTicker) -> bool:
        """Save ticker information to database.
        
        Args:
            ticker_info: Ticker information to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                # Use INSERT OR REPLACE for idempotent operation
                ticker_record = Ticker(
                    ticker=ticker_info.ticker,
                    name=ticker_info.name,
                    exchange=ticker_info.exchange,
                    country=ticker_info.country,
                    sector=ticker_info.sector,
                    industry=ticker_info.industry,
                    first_listed=ticker_info.first_listed,
                    last_updated=ticker_info.last_updated
                )
                
                # Use merge for idempotent operation
                session.merge(ticker_record)
                session.commit()
                
                logger.info(f"Saved ticker info for {ticker_info.ticker}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save ticker info for {ticker_info.ticker}: {str(e)}")
            return False
    
    def save_daily_metrics(self, metrics_list: List[DatabaseDailyMetrics]) -> int:
        """Save daily metrics to database with idempotent operations.
        
        Args:
            metrics_list: List of daily metrics to save
            
        Returns:
            Number of records saved
        """
        if not metrics_list:
            return 0
        
        saved_count = 0
        
        try:
            with self.get_session() as session:
                for metrics in metrics_list:
                    try:
                        # Create record
                        daily_record = DailyMetrics(
                            ticker=metrics.ticker,
                            date=metrics.date,
                            open=float(metrics.open),
                            high=float(metrics.high),
                            low=float(metrics.low),
                            close=float(metrics.close),
                            volume=metrics.volume,
                            sma_50=float(metrics.sma_50) if metrics.sma_50 else None,
                            sma_200=float(metrics.sma_200) if metrics.sma_200 else None,
                            high_52w=float(metrics.high_52w) if metrics.high_52w else None,
                            pct_from_52w_high=float(metrics.pct_from_52w_high) if metrics.pct_from_52w_high else None,
                            price_to_book=float(metrics.price_to_book) if metrics.price_to_book else None,
                            book_value_per_share=float(metrics.book_value_per_share) if metrics.book_value_per_share else None,
                            enterprise_value=float(metrics.enterprise_value) if metrics.enterprise_value else None
                        )
                        
                        # Use merge for idempotent operation
                        session.merge(daily_record)
                        saved_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to save daily metrics for {metrics.ticker} on {metrics.date}: {str(e)}")
                        continue
                
                session.commit()
                logger.info(f"Saved {saved_count} daily metrics records")
                
        except Exception as e:
            logger.error(f"Failed to save daily metrics: {str(e)}")
        
        return saved_count
    
    def save_signal_events(self, signal_events: List[DatabaseSignalEvent]) -> int:
        """Save signal events to database.
        
        Args:
            signal_events: List of signal events to save
            
        Returns:
            Number of records saved
        """
        if not signal_events:
            return 0
        
        saved_count = 0
        
        try:
            with self.get_session() as session:
                for signal in signal_events:
                    try:
                        # Create record
                        signal_record = SignalEvents(
                            ticker=signal.ticker,
                            signal_type=signal.signal_type,
                            date=signal.date,
                            sma_50=float(signal.sma_50),
                            sma_200=float(signal.sma_200),
                            price=float(signal.price),
                            volume=signal.volume,
                            metadata=signal.metadata,
                            created_at=signal.created_at
                        )
                        
                        session.add(signal_record)
                        saved_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to save signal event for {signal.ticker} on {signal.date}: {str(e)}")
                        continue
                
                session.commit()
                logger.info(f"Saved {saved_count} signal events")
                
        except Exception as e:
            logger.error(f"Failed to save signal events: {str(e)}")
        
        return saved_count
    
    def save_processed_data(self, processed_data: ProcessedDataFrame) -> Dict[str, int]:
        """Save processed data to database.
        
        Args:
            processed_data: ProcessedDataFrame to save
            
        Returns:
            Dictionary with save results
        """
        results = {
            "ticker_info_saved": 0,
            "daily_metrics_saved": 0,
            "signal_events_saved": 0
        }
        
        try:
            # Save ticker info
            ticker_info = DatabaseTicker(
                ticker=processed_data.ticker,
                name=processed_data.processing_metadata.get("ticker_name"),
                exchange=processed_data.processing_metadata.get("exchange"),
                country=processed_data.processing_metadata.get("country"),
                last_updated=datetime.utcnow()
            )
            
            if self.save_ticker_info(ticker_info):
                results["ticker_info_saved"] = 1
            
            # Convert and save daily metrics
            daily_metrics = self._convert_to_daily_metrics(processed_data)
            results["daily_metrics_saved"] = self.save_daily_metrics(daily_metrics)
            
            # Convert and save signal events
            signal_events = self._convert_to_signal_events(processed_data.signal_events)
            results["signal_events_saved"] = self.save_signal_events(signal_events)
            
            logger.info(f"Saved processed data for {processed_data.ticker}: {results}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
        
        return results
    
    def _convert_to_daily_metrics(self, processed_data: ProcessedDataFrame) -> List[DatabaseDailyMetrics]:
        """Convert processed data to daily metrics format.
        
        Args:
            processed_data: ProcessedDataFrame
            
        Returns:
            List of DatabaseDailyMetrics objects
        """
        daily_metrics = []
        
        for record in processed_data.data:
            try:
                metrics = DatabaseDailyMetrics(
                    ticker=processed_data.ticker,
                    date=datetime.fromisoformat(record['date']).date() if isinstance(record['date'], str) else record['date'],
                    open=record['open'],
                    high=record['high'],
                    low=record['low'],
                    close=record['close'],
                    volume=record['volume'],
                    sma_50=record.get('sma_50'),
                    sma_200=record.get('sma_200'),
                    high_52w=record.get('high_52w'),
                    pct_from_52w_high=record.get('pct_from_52w_high'),
                    price_to_book=record.get('price_to_book'),
                    book_value_per_share=record.get('book_value_per_share'),
                    enterprise_value=record.get('enterprise_value')
                )
                daily_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to convert daily metrics record: {str(e)}")
                continue
        
        return daily_metrics
    
    def _convert_to_signal_events(self, signal_events: List[SignalEvent]) -> List[DatabaseSignalEvent]:
        """Convert signal events to database format.
        
        Args:
            signal_events: List of SignalEvent objects
            
        Returns:
            List of DatabaseSignalEvent objects
        """
        db_events = []
        
        for signal in signal_events:
            try:
                db_event = DatabaseSignalEvent(
                    ticker=signal.ticker,
                    signal_type=signal.signal_type,
                    date=signal.date,
                    sma_50=signal.sma_50,
                    sma_200=signal.sma_200,
                    price=signal.price,
                    volume=signal.volume,
                    metadata=json.dumps(signal.metadata),
                    created_at=datetime.utcnow()
                )
                db_events.append(db_event)
                
            except Exception as e:
                logger.warning(f"Failed to convert signal event: {str(e)}")
                continue
        
        return db_events
    
    def get_ticker_data(self, ticker: str, start_date: Optional[date] = None, 
                       end_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """Retrieve ticker data from database.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            List of data dictionaries
        """
        try:
            with self.get_session() as session:
                query = session.query(DailyMetrics).filter(DailyMetrics.ticker == ticker)
                
                if start_date:
                    query = query.filter(DailyMetrics.date >= start_date)
                if end_date:
                    query = query.filter(DailyMetrics.date <= end_date)
                
                results = query.order_by(DailyMetrics.date).all()
                
                data = []
                for record in results:
                    data.append({
                        'date': record.date.isoformat(),
                        'open': record.open,
                        'high': record.high,
                        'low': record.low,
                        'close': record.close,
                        'volume': record.volume,
                        'sma_50': record.sma_50,
                        'sma_200': record.sma_200,
                        'high_52w': record.high_52w,
                        'pct_from_52w_high': record.pct_from_52w_high,
                        'price_to_book': record.price_to_book,
                        'book_value_per_share': record.book_value_per_share,
                        'enterprise_value': record.enterprise_value
                    })
                
                logger.info(f"Retrieved {len(data)} records for {ticker}")
                return data
                
        except Exception as e:
            logger.error(f"Failed to retrieve data for {ticker}: {str(e)}")
            return []
    
    def get_signal_events(self, ticker: str, signal_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve signal events from database.
        
        Args:
            ticker: Ticker symbol
            signal_type: Optional signal type filter
            
        Returns:
            List of signal event dictionaries
        """
        try:
            with self.get_session() as session:
                query = session.query(SignalEvents).filter(SignalEvents.ticker == ticker)
                
                if signal_type:
                    query = query.filter(SignalEvents.signal_type == signal_type)
                
                results = query.order_by(SignalEvents.date).all()
                
                events = []
                for record in results:
                    events.append({
                        'ticker': record.ticker,
                        'signal_type': record.signal_type,
                        'date': record.date.isoformat(),
                        'sma_50': record.sma_50,
                        'sma_200': record.sma_200,
                        'price': record.price,
                        'volume': record.volume,
                        'metadata': json.loads(record.metadata) if record.metadata else {},
                        'created_at': record.created_at.isoformat()
                    })
                
                logger.info(f"Retrieved {len(events)} signal events for {ticker}")
                return events
                
        except Exception as e:
            logger.error(f"Failed to retrieve signal events for {ticker}: {str(e)}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with self.get_session() as session:
                ticker_count = session.query(Ticker).count()
                daily_metrics_count = session.query(DailyMetrics).count()
                signal_events_count = session.query(SignalEvents).count()
                
                return {
                    "tickers": ticker_count,
                    "daily_metrics": daily_metrics_count,
                    "signal_events": signal_events_count,
                    "database_path": self.database_path
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {}
