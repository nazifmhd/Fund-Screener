# Fund-Screener: Production-Grade Financial Analysis Pipeline

A comprehensive financial analysis pipeline that processes stock data, calculates technical indicators, detects trading signals, and provides detailed analysis for both US and Indian markets, including recent IPOs.

## 🚀 Features

- **Multi-Market Support**: Handles both US and Indian stock markets
- **Recent IPO Support**: Works with stocks as new as 10 months old
- **Technical Analysis**: Calculates 50-day and 200-day SMAs, 52-week highs
- **Signal Detection**: Detects Golden Crossover and Death Cross patterns
- **Fundamental Analysis**: Processes financial ratios and metrics
- **Data Quality Handling**: Robust error handling and data validation
- **Database Storage**: SQLite database with proper schema design
- **CLI Interface**: Easy-to-use command-line interface
- **JSON Export**: Structured data export for further analysis

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Database Schema](#database-schema)
- [Design Decisions](#design-decisions)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🛠 Installation

### Prerequisites

- Python 3.9 or higher
- pip or uv package manager

### Using pip

```bash
# Clone the repository
git clone [<repository-url>](https://github.com/nazifmhd/Fund-Screener.git)
cd Fund-Screener

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Using uv (Recommended)

```bash
# Clone the repository
git clone [<repository-url>](https://github.com/nazifmhd/Fund-Screener.git)
cd Fund-Screener

# Install dependencies
uv sync

# Install development dependencies
uv sync --dev
```

## 🚀 Quick Start

1. **Copy configuration file**:
   ```bash
   cp config.yaml.example config.yaml
   ```

2. **Run your first analysis**:
   ```bash
   # US Stock
   python -m src.main --ticker NVDA --output nvda_analysis.json
   
   # Indian Stock
   python -m src.main --ticker RELIANCE.NS --output reliance_analysis.json
   
   # Recent IPO
   python -m src.main --ticker SWIGGY.NS --output swiggy_analysis.json
   ```

3. **View results**:
   The analysis results are saved as JSON files with comprehensive data including:
   - Price history with technical indicators
   - Trading signals (Golden Crossover/Death Cross)
   - Fundamental metrics
   - Data quality notes

## 📖 Usage Examples

### Basic Analysis

```bash
# Analyze NVIDIA (US stock)
python -m src.main --ticker NVDA --output nvda_analysis.json

# Analyze Reliance Industries (Indian stock)
python -m src.main --ticker RELIANCE.NS --output reliance_analysis.json

# Analyze recent IPO
python -m src.main --ticker SWIGGY.NS --output swiggy_analysis.json
```

### Advanced Usage

```bash
# Verbose logging
python -m src.main --ticker NVDA --output nvda_analysis.json --verbose

# Custom configuration
python -m src.main --ticker NVDA --output nvda_analysis.json --config custom_config.yaml

# Test pipeline with multiple tickers
python -m src.main test-pipeline --verbose
```

### Database Operations

```bash
# View database statistics
python -m src.main database-stats
```

## 🏗 Architecture

The pipeline follows a modular architecture with clear separation of concerns:

```
src/
├── models.py          # Pydantic schemas for data validation
├── data_fetcher.py    # yfinance API integration
├── processor.py       # Data processing and calculations
├── signals.py         # Trading signal detection
├── database.py        # SQLite operations
├── config.py          # Configuration management
└── main.py           # CLI interface
```

### Data Flow

1. **Data Ingestion**: Fetch data from yfinance API
2. **Validation**: Validate data using Pydantic schemas
3. **Processing**: Calculate technical indicators and fundamental ratios
4. **Signal Detection**: Detect trading signals (Golden Crossover/Death Cross)
5. **Storage**: Save to SQLite database
6. **Export**: Generate JSON analysis report

## 🗄 Database Schema

The application uses SQLite with three main tables:

### `tickers` Table
- `ticker` (PRIMARY KEY): Stock symbol
- `name`: Company name
- `exchange`: Stock exchange
- `country`: Country of listing
- `sector`: Business sector
- `industry`: Industry classification
- `first_listed`: First listing date
- `last_updated`: Last update timestamp

### `daily_metrics` Table
- `id` (PRIMARY KEY): Auto-increment ID
- `ticker`: Stock symbol (with UNIQUE constraint on ticker+date)
- `date`: Trading date
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume
- `sma_50`, `sma_200`: Moving averages
- `high_52w`: 52-week high
- `pct_from_52w_high`: Percentage from 52-week high
- `price_to_book`: Price-to-book ratio
- `book_value_per_share`: Book value per share
- `enterprise_value`: Enterprise value

### `signal_events` Table
- `id` (PRIMARY KEY): Auto-increment ID
- `ticker`: Stock symbol
- `signal_type`: Type of signal (golden_crossover/death_cross)
- `date`: Signal date
- `sma_50`, `sma_200`: SMA values at signal
- `price`: Stock price at signal
- `volume`: Volume at signal
- `metadata`: Additional signal metadata (JSON)
- `created_at`: Signal detection timestamp

## 🎯 Design Decisions

### 1. Frequency Mismatch Problem

**Problem**: Stock prices update daily, but financial statements only update quarterly.

**Solution**: 
- Use forward-fill strategy for fundamental data
- Document data quality issues when using forward-filled data
- Provide synthetic metrics from info data when balance sheet data is unavailable

**Trade-offs**:
- ✅ Maintains data continuity
- ✅ Provides reasonable estimates for recent periods
- ⚠️ May not reflect recent fundamental changes

### 2. Unreliable Fundamental Data

**Problem**: yfinance fundamental data is often missing or incomplete.

**Solution**:
- **Primary**: Try quarterly balance sheet data
- **Fallback 1**: Use annual balance sheet data
- **Fallback 2**: Create synthetic metrics from info data
- **Documentation**: Log all data quality issues

**Implementation**:
```python
# Strategy 1: Quarterly data
quarterly_bs = ticker.quarterly_balance_sheet

# Strategy 2: Annual fallback
if quarterly_bs.empty:
    annual_bs = ticker.balance_sheet

# Strategy 3: Synthetic data
if annual_bs.empty:
    synthetic_data = create_synthetic_fundamental_data(ticker.info)
```

### 3. Golden Crossover Detection

**Logic**: Detect when 50-day SMA crosses above 200-day SMA.

**Implementation**:
- Use vectorized pandas operations for efficiency
- Handle edge cases (insufficient data, NaN values)
- Return list of crossover dates with metadata

**Edge Cases**:
- Insufficient data (< 200 trading days)
- NaN values in SMA calculations
- Multiple crossovers in same dataset

### 4. Multi-Market & Recent Stock Handling

**Challenge**: Handle stocks from different markets with varying data availability.

**Solution**:
- **Ticker Format Detection**: Automatically detect market based on suffix
  - US stocks: No suffix (e.g., `NVDA`)
  - Indian stocks: `.NS` suffix (e.g., `RELIANCE.NS`)
- **Data Availability Adaptation**: Gracefully handle limited history
- **Market-Specific Handling**: Different strategies for different markets

**Recent IPO Handling**:
- Accept stocks with as little as 10 months of data
- Adjust SMA calculations based on available data
- Provide clear warnings about data limitations

### 5. Database Design

**Idempotent Operations**: Use `INSERT OR REPLACE` to prevent duplicates
**Unique Constraints**: Ensure data integrity with ticker+date combinations
**Schema Design**: Normalized structure with proper relationships

### 6. Error Handling Strategy

**Graceful Degradation**: Continue processing with partial data when possible
**Comprehensive Logging**: Log all data quality issues and processing steps
**User-Friendly Messages**: Provide clear error messages and suggestions

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_processor.py

# Run with verbose output
pytest -v
```

### Test Coverage

The test suite covers:
- **Data Processing**: Metric calculations and data transformations
- **Signal Detection**: Golden Crossover and Death Cross detection
- **Data Validation**: Pydantic model validation
- **Error Handling**: API failures and data quality issues
- **Database Operations**: CRUD operations and data integrity

### Test Data

Tests use realistic sample data including:
- Price data with OHLCV information
- Fundamental data with balance sheet metrics
- Signal events with proper metadata
- Edge cases for data quality issues

## ⚙️ Configuration

### Configuration File (`config.yaml`)

```yaml
database:
  path: "financial_data.db"

logging:
  level: "INFO"

data_settings:
  historical_period: "5y"
  min_trading_days_for_sma: 200
  fallback_fundamental_data: true
  forward_fill_fundamentals: true

api_settings:
  timeout: 30
  retry_attempts: 3
  retry_delay: 1

markets:
  us:
    suffix: ""
    exchange: "US"
  india:
    suffix: ".NS"
    exchange: "NSE"
```

### Environment Variables

```bash
# Optional: Override database path
export FUND_SCREENER_DB_PATH="/path/to/database.db"

# Optional: Override log level
export FUND_SCREENER_LOG_LEVEL="DEBUG"
```

## 🔧 Troubleshooting

### Common Issues

1. **"No price data available"**
   - Check if ticker symbol is correct
   - Verify market is open (for real-time data)
   - Try with different historical period

2. **"Insufficient data for SMA calculation"**
   - Normal for recent IPOs
   - Pipeline will continue with available data
   - SMA calculations will be NaN for insufficient data

3. **"Missing fundamental data"**
   - Check data quality notes in output
   - Pipeline uses fallback strategies
   - Synthetic data may be used

4. **Database connection issues**
   - Check database file permissions
   - Ensure SQLite is properly installed
   - Verify database path in configuration

### Debug Mode

```bash
# Enable verbose logging
python -m src.main --ticker NVDA --output nvda_analysis.json --verbose

# Check logs
tail -f fund_screener.log
```

### Data Quality Issues

The pipeline provides comprehensive data quality reporting:
- Missing data percentages
- Fallback strategy usage
- Data source documentation
- Processing metadata

## 📊 Output Format

### JSON Export Structure

```json
{
  "ticker": "NVDA",
  "analysis_date": "2024-01-15T10:30:00",
  "data_period": {
    "start_date": "2019-01-15",
    "end_date": "2024-01-15",
    "total_days": 1250
  },
  "technical_indicators": [
    {
      "date": "2024-01-15",
      "sma_50": 450.25,
      "sma_200": 420.10,
      "high_52w": 500.00,
      "pct_from_52w_high": -9.95
    }
  ],
  "signal_events": [
    {
      "ticker": "NVDA",
      "signal_type": "golden_crossover",
      "date": "2024-01-10",
      "sma_50": 445.50,
      "sma_200": 440.25,
      "price": 448.75,
      "volume": 25000000
    }
  ],
  "data_quality_notes": [
    "Using annual data as quarterly fallback",
    "Forward-fill applied to fundamental data"
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- yfinance for financial data API
- SQLAlchemy for database operations
- Typer for CLI interface
- Pydantic for data validation
- Rich for beautiful console output
