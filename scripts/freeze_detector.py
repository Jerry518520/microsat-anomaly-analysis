"""Explicit offline package build. Never invoked by API or detect()."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection.frozen import DEFAULT_BUNDLE, freeze_bundle, load_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        manifest = freeze_bundle(args.output)
        print(f"built {manifest['detector_version']}; fit rows={manifest['fit_rows']}")
    judges = load_bundle(args.output)
    print(f"verified {args.output}; channels={len(judges)}; no inference training")


if __name__ == "__main__":
    main()
