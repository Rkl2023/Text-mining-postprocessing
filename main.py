import logging
import sys
import json
from pathlib import Path

from polymer_pipeline.cli import build_parser
from polymer_pipeline.io_manager import detect_input_files
from polymer_pipeline.logging_utils import setup_logger
from polymer_pipeline.pipeline import run_pipeline


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

    report_path = Path("logs/autoload_report.json")
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            summary = report.get("summary", {})
            print("Detected polymer file:", summary.get("selected_polymer_file"))
            print("Detected image pool file:", summary.get("selected_image_file"))
        except Exception:
            pass

    run_pipeline(text_path, image_path, output_path=args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
