"""
Download NYFed DSGE Model 1002 Data from FRED

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
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsge.data import load_nyfed_data, validate_data


def main():
    """Main data download routine."""
    parser = argparse.ArgumentParser(description='Download NYFed DSGE data from FRED')

    parser.add_argument(
        '--start',
        type=str,
        default='1960-01-01',
        help='Start date (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD format, default: today)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='FRED API key (or set FRED_API_KEY environment variable)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/nyfed_data.csv',
        help='Output CSV file path'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run data validation after download'
    )

    args = parser.parse_args()

    # Check for API key
    api_key = args.api_key or os.environ.get('FRED_API_KEY')

    if api_key is None:
        print("="*80)
        print("ERROR: FRED API key required")
        print("="*80)
        print("\nTo use this script, you need a free FRED API key:")
        print("1. Go to https://fred.stlouisfed.org/")
        print("2. Create a free account")
        print("3. Get API key at https://fred.stlouisfed.org/docs/api/api_key.html")
        print("\nThen either:")
        print("  - Set environment variable: export FRED_API_KEY=your_key")
        print("  - Pass as argument: --api-key your_key")
        print("\n" + "="*80)
        sys.exit(1)

    print("="*80)
    print("NYFed DSGE Model 1002 - Data Download")
    print("="*80)
    print(f"\nStart date: {args.start}")
    print(f"End date: {args.end or 'present'}")
    print(f"Output file: {args.output}")

    # Download data
    try:
        data = load_nyfed_data(
            start_date=args.start,
            end_date=args.end,
            api_key=api_key,
            save_path=args.output
        )

        # Validate if requested
        if args.validate:
            print("\n" + "="*80)
            print("Running data validation...")
            print("="*80)
            validate_data(data, verbose=True)

        print("\n" + "="*80)
        print("✅ Data download complete!")
        print("="*80)
        print(f"\nData saved to: {args.output}")
        print(f"Observations: {len(data)}")
        print(f"Variables: {len(data.columns)}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")

    except Exception as e:
        print("\n" + "="*80)
        print("❌ Error during data download:")
        print("="*80)
        print(f"\n{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
