"""Command-line entry; see README for configuration and usage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from convnextv2_omics.cli import cv_main

if __name__ == "__main__":
    cv_main()
