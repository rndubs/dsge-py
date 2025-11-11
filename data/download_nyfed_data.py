"""
Download NYFed DSGE Model 1002 Data from FRED.

This script downloads all 13 observable variables for the NYFed model
from FRED and saves the processed data to a CSV file.

Requirements:
- FRED API key (free from https://fred.stlouisfed.org/docs/api/api_key.html)
- Set as environment variable: export FRED_API_KEY=your_key

Usage:
    python data/download_nyfed_data.py
    python data/download_nyfed_data.py --start 1960-01-01 --end 2024-01-01
    python data/download_nyfed_data.py --api-key YOUR_KEY
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsge.data import load_nyfed_data, validate_data


def main() -> None:
    """Main data download routine."""
    parser = argparse.ArgumentParser(description="Download NYFed DSGE data from FRED")

    parser.add_argument(
        "--start", type=str, default="1960-01-01", help="Start date (YYYY-MM-DD format)"
    )

    parser.add_argument(
        "--end", type=str, default=None, help="End date (YYYY-MM-DD format, default: today)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="FRED API key (or set FRED_API_KEY environment variable)",
    )

    parser.add_argument(
        "--output", type=str, default="data/nyfed_data.csv", help="Output CSV file path"
    )

    parser.add_argument(
        "--validate", action="store_true", help="Run data validation after download"
    )

    args = parser.parse_args()

    # Check for API key
    api_key = args.api_key or os.environ.get("FRED_API_KEY")

    if api_key is None:
        sys.exit(1)

    # Download data
    try:
        data = load_nyfed_data(
            start_date=args.start, end_date=args.end, api_key=api_key, save_path=args.output
        )

        # Validate if requested
        if args.validate:
            validate_data(data, verbose=True)

    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
