"""Main CLI interface for the financial analysis pipeline."""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config
from .data_fetcher import DataFetcher
from .processor import DataProcessor
from .signals import SignalDetector
from .database import DatabaseManager
from .models import ExportData


# Initialize Typer app
app = typer.Typer(
    name="fund-screener",
    help="Production-grade financial analysis pipeline",
    add_completion=False
)

# Initialize console for rich output
console = Console()


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fund_screener.log')
        ]
    )


@app.command()
def analyze(
    ticker: str = typer.Option(..., "-t", "--ticker", help="Stock ticker symbol (e.g., NVDA, RELIANCE.NS)"),
    output: str = typer.Option(..., "-o", "--output", help="Output JSON file path"),
    config_path: Optional[str] = typer.Option(None, "-c", "--config", help="Path to configuration file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging")
):
    """Analyze a stock ticker and generate comprehensive financial analysis.
    
    Examples:
        # US stock
        python -m src.main --ticker NVDA --output nvda_analysis.json
        
        # Indian stock  
        python -m src.main --ticker RELIANCE.NS --output reliance_analysis.json
        
        # Recent IPO
        python -m src.main --ticker SWIGGY.NS --output swiggy_analysis.json
    """
    try:
        # Setup logging
        log_level = "DEBUG" if verbose else "INFO"
        setup_logging(log_level)
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting analysis for ticker: {ticker}")
        
        # Load configuration
        config = Config(config_path)
        
        # Validate ticker format
        if not config.is_valid_ticker(ticker):
            console.print(f"[red]Error: Invalid ticker format: {ticker}[/red]")
            console.print("Supported formats: NVDA (US), RELIANCE.NS (India)")
            raise typer.Exit(1)
        
        # Initialize components
        data_fetcher = DataFetcher(
            timeout=config.api_timeout,
            retry_attempts=config.api_retry_attempts,
            retry_delay=config.api_retry_delay
        )
        
        data_processor = DataProcessor(
            min_trading_days_for_sma=config.min_trading_days_for_sma,
            forward_fill_fundamentals=config.forward_fill_fundamentals
        )
        
        signal_detector = SignalDetector(min_data_points=config.min_trading_days_for_sma)
        
        database_manager = DatabaseManager(config.database_path)
        
        # Execute analysis pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: Fetch data
            task1 = progress.add_task("Fetching stock data...", total=None)
            try:
                raw_data = data_fetcher.fetch_stock_data(ticker, config.historical_period)
                progress.update(task1, description="✅ Data fetched successfully")
            except Exception as e:
                progress.update(task1, description="❌ Data fetch failed")
                console.print(f"[red]Error fetching data: {str(e)}[/red]")
                raise typer.Exit(1)
            
            # Step 2: Process data
            task2 = progress.add_task("Processing data and calculating indicators...", total=None)
            try:
                processed_data = data_processor.process_data(raw_data)
                progress.update(task2, description="✅ Data processed successfully")
            except Exception as e:
                progress.update(task2, description="❌ Data processing failed")
                console.print(f"[red]Error processing data: {str(e)}[/red]")
                raise typer.Exit(1)
            
            # Step 3: Detect signals
            task3 = progress.add_task("Detecting trading signals...", total=None)
            try:
                processed_data = signal_detector.detect_all_signals(processed_data)
                progress.update(task3, description="✅ Signal detection completed")
            except Exception as e:
                progress.update(task3, description="❌ Signal detection failed")
                console.print(f"[red]Error detecting signals: {str(e)}[/red]")
                raise typer.Exit(1)
            
            # Step 4: Save to database
            task4 = progress.add_task("Saving to database...", total=None)
            try:
                save_results = database_manager.save_processed_data(processed_data)
                progress.update(task4, description="✅ Database save completed")
            except Exception as e:
                progress.update(task4, description="❌ Database save failed")
                console.print(f"[red]Error saving to database: {str(e)}[/red]")
                # Continue execution even if database save fails
        
        # Step 5: Generate export data
        export_data = create_export_data(processed_data, ticker)
        
        # Step 6: Save to JSON file
        try:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(export_data.model_dump(), f, indent=2, default=str)
            
            console.print(f"[green]✅ Analysis completed successfully![/green]")
            console.print(f"[green]Results saved to: {output_path.absolute()}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error saving output file: {str(e)}[/red]")
            raise typer.Exit(1)
        
        # Display summary
        display_analysis_summary(export_data, save_results)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def create_export_data(processed_data, ticker: str) -> ExportData:
    """Create export data from processed data.
    
    Args:
        processed_data: ProcessedDataFrame object
        ticker: Stock ticker symbol
        
    Returns:
        ExportData object for JSON export
    """
    # Extract technical indicators
    technical_indicators = []
    for record in processed_data.data:
        if any(record.get(col) is not None for col in ['sma_50', 'sma_200', 'high_52w']):
            technical_indicators.append({
                'date': record['date'],
                'sma_50': record.get('sma_50'),
                'sma_200': record.get('sma_200'),
                'high_52w': record.get('high_52w'),
                'pct_from_52w_high': record.get('pct_from_52w_high'),
                'price_to_book': record.get('price_to_book'),
                'book_value_per_share': record.get('book_value_per_share'),
                'enterprise_value': record.get('enterprise_value')
            })
    
    # Get data period
    if processed_data.data:
        dates = [record['date'] for record in processed_data.data]
        data_period = {
            'start_date': min(dates),
            'end_date': max(dates),
            'total_days': len(dates)
        }
    else:
        data_period = {'start_date': None, 'end_date': None, 'total_days': 0}
    
    return ExportData(
        ticker=ticker,
        analysis_date=datetime.now(),
        data_period=data_period,
        technical_indicators=technical_indicators,
        signal_events=processed_data.signal_events,
        fundamental_metrics=processed_data.processing_metadata,
        data_quality_notes=processed_data.data_quality_notes,
        processing_metadata=processed_data.processing_metadata
    )


def display_analysis_summary(export_data: ExportData, save_results: dict):
    """Display analysis summary in a formatted table.
    
    Args:
        export_data: Export data object
        save_results: Database save results
    """
    console.print("\n[bold blue]Analysis Summary[/bold blue]")
    
    # Create summary table
    table = Table(title=f"Analysis Results for {export_data.ticker}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Analysis Date", export_data.analysis_date.strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("Data Period", f"{export_data.data_period['start_date']} to {export_data.data_period['end_date']}")
    table.add_row("Total Days", str(export_data.data_period['total_days']))
    table.add_row("Technical Indicators", str(len(export_data.technical_indicators)))
    table.add_row("Signal Events", str(len(export_data.signal_events)))
    
    # Add signal breakdown
    if export_data.signal_events:
        signal_types = {}
        for signal in export_data.signal_events:
            signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1
        
        for signal_type, count in signal_types.items():
            table.add_row(f"  {signal_type.replace('_', ' ').title()}", str(count))
    
    # Add database save results
    if save_results:
        table.add_row("Database Records", f"Ticker: {save_results.get('ticker_info_saved', 0)}, "
                                        f"Metrics: {save_results.get('daily_metrics_saved', 0)}, "
                                        f"Signals: {save_results.get('signal_events_saved', 0)}")
    
    # Add data quality notes
    if export_data.data_quality_notes:
        table.add_row("Data Quality Issues", str(len(export_data.data_quality_notes)))
    
    console.print(table)
    
    # Display recent signals if any
    if export_data.signal_events:
        console.print("\n[bold yellow]Recent Trading Signals[/bold yellow]")
        signal_table = Table()
        signal_table.add_column("Date", style="cyan")
        signal_table.add_column("Type", style="green")
        signal_table.add_column("Price", style="yellow")
        signal_table.add_column("SMA 50", style="blue")
        signal_table.add_column("SMA 200", style="blue")
        
        # Show last 5 signals
        recent_signals = sorted(export_data.signal_events, key=lambda x: x.date, reverse=True)[:5]
        for signal in recent_signals:
            signal_table.add_row(
                signal.date.strftime("%Y-%m-%d"),
                signal.signal_type.replace('_', ' ').title(),
                f"${signal.price:.2f}",
                f"{signal.sma_50:.2f}",
                f"{signal.sma_200:.2f}"
            )
        
        console.print(signal_table)


@app.command()
def test_pipeline(
    config_path: Optional[str] = typer.Option(None, "-c", "--config", help="Path to configuration file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging")
):
    """Test the pipeline with sample tickers from different markets.
    
    This command tests the pipeline with:
    - US stocks (NVDA, AAPL)
    - Indian stocks (RELIANCE.NS, TCS.NS)  
    - Recent IPOs (SWIGGY.NS, HYUNDAI.NS)
    """
    test_tickers = [
        ("NVDA", "US Stock"),
        ("AAPL", "US Stock"),
        ("RELIANCE.NS", "Indian Stock"),
        ("TCS.NS", "Indian Stock"),
        ("SWIGGY.NS", "Recent IPO (India)"),
        ("HYUNDAI.NS", "Recent IPO (India)")
    ]
    
    console.print("[bold blue]Testing Pipeline with Multiple Tickers[/bold blue]")
    
    for ticker, description in test_tickers:
        console.print(f"\n[cyan]Testing {ticker} ({description})[/cyan]")
        
        try:
            # Run analysis for each ticker
            output_file = f"test_{ticker.replace('.', '_').lower()}_analysis.json"
            
            # Call analyze function directly
            analyze(
                ticker=ticker,
                output=output_file,
                config_path=config_path,
                verbose=verbose
            )
            
            console.print(f"[green]✅ {ticker} analysis completed[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ {ticker} analysis failed: {str(e)}[/red]")
            continue
    
    console.print("\n[bold green]Pipeline testing completed![/bold green]")


@app.command()
def database_stats(
    config_path: Optional[str] = typer.Option(None, "-c", "--config", help="Path to configuration file")
):
    """Display database statistics and contents."""
    try:
        config = Config(config_path)
        db_manager = DatabaseManager(config.database_path)
        
        stats = db_manager.get_database_stats()
        
        console.print("[bold blue]Database Statistics[/bold blue]")
        
        table = Table(title="Database Contents")
        table.add_column("Table", style="cyan")
        table.add_column("Records", style="green")
        
        table.add_row("Tickers", str(stats.get('tickers', 0)))
        table.add_row("Daily Metrics", str(stats.get('daily_metrics', 0)))
        table.add_row("Signal Events", str(stats.get('signal_events', 0)))
        
        console.print(table)
        console.print(f"\nDatabase location: {stats.get('database_path', 'Unknown')}")
        
    except Exception as e:
        console.print(f"[red]Error getting database stats: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
