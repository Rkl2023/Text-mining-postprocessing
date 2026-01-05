import argparse
import logging
import sys

from .io_manager import detect_input_files
from .logging_utils import setup_logger
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polymer structure mining pipeline")
    parser.add_argument(
        "--text_table",
        "--polymer_data",
        dest="text_table",
        required=False,
        help="Path to text-mined polymer table (CSV/JSON/Parquet/SQLite). If omitted, auto-detection scans ./data",
    )
    parser.add_argument(
        "--image_pool",
        dest="image_pool",
        required=False,
        help="Path to image-recognition candidate pool. If omitted, auto-detection scans ./data",
    )
    parser.add_argument("--output", default="polymer_structures_groundtruth.csv", help="Output CSV path")
    parser.add_argument("--log_file", default="logs/pipeline.log", help="Log file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--fast_mode", action="store_true", help="Skip expensive retries/sanitization for speed")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(log_file=args.log_file, level=level)

    text_path = args.text_table
    image_path = args.image_pool

    if not text_path or not image_path:
        auto_polymer, auto_image = detect_input_files(data_dir="data", autoload_log="logs/autoload_report.json")
        if not text_path:
            text_path = str(auto_polymer) if auto_polymer else None
        if not image_path:
            image_path = str(auto_image) if auto_image else None

    if not text_path or not image_path:
        print("No valid input files detected in /data. Please add polymer and image candidate files.", file=sys.stderr)
        sys.exit(1)

    run_pipeline(text_path, image_path, output_path=args.output, fast_mode=args.fast_mode)


if __name__ == "__main__":  # pragma: no cover
    main()
