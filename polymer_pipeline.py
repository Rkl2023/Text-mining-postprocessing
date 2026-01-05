"""
One-shot launcher for the PolyStruct-Mine pipeline with sensible defaults.

Usage:
    python polymer_pipeline.py
"""

import argparse
import os
import sys
from pathlib import Path

from polymer_pipeline.io_manager import detect_input_files
from polymer_pipeline.logging_utils import setup_logger
from polymer_pipeline.pipeline import run_pipeline

DEFAULT_TEXT_TABLE = "examples/text_polymers_sample.csv"
DEFAULT_IMAGE_POOL = "examples/image_pool_sample.csv"
DEFAULT_OUTPUT = "outputs/polymer_structures_groundtruth.csv"
DEFAULT_LOG = "logs/pipeline.log"


def main():
    parser = argparse.ArgumentParser(description="Run PolyStruct-Mine with auto-detected inputs.")
    parser.add_argument("--text_table", "--polymer_data", dest="text_table", default=None, help="Optional override for polymer/text dataset")
    parser.add_argument("--image_pool", dest="image_pool", default=None, help="Optional override for image/candidate dataset")
    parser.add_argument("--output", dest="output", default=None, help="Optional override for output CSV path")
    parser.add_argument("--log_file", dest="log_file", default=None, help="Optional override for log file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--fast_mode", action="store_true", help="Skip expensive retries/sanitization for speed")
    args = parser.parse_args()

    text_table = args.text_table or DEFAULT_TEXT_TABLE
    image_pool = args.image_pool or DEFAULT_IMAGE_POOL
    output = args.output or DEFAULT_OUTPUT
    log_file = args.log_file or DEFAULT_LOG

    # If defaults do not exist, try schema-based detection from /data.
    if not Path(text_table).exists() or not Path(image_pool).exists():
        print("🔍 Auto-detecting input files in data/ or /data ...")
        detected_polymer, detected_image = detect_input_files(data_dir="data", autoload_log="logs/autoload_report.json")
        text_table = str(detected_polymer) if detected_polymer else text_table
        image_pool = str(detected_image) if detected_image else image_pool

    # Final existence check.
    if not Path(text_table).exists() or not Path(image_pool).exists():
        print("No valid input files detected. Provide --text_table and --image_pool or add files under data/.", file=sys.stderr)
        sys.exit(1)

    # Ensure output/log directories exist.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    setup_logger(log_file=log_file, level=10 if args.debug else 20)

    print("🧪 Running PolyStruct-Mine pipeline with:")
    print(f"  Text-mined table: {text_table}")
    print(f"  Image pool:       {image_pool}")
    print(f"  Output CSV:       {output}")
    print(f"  Log file:         {log_file}")
    print(f"  Debug mode:       {args.debug}")
    print(f"  Fast mode:        {args.fast_mode}")

    run_pipeline(text_table, image_pool, output_path=output, fast_mode=args.fast_mode)


if __name__ == "__main__":  # pragma: no cover
    main()
