"""
Download DexiNed pretrained weights (BIPED dataset).

Usage:
    python scripts/download_dexined.py [--output checkpoints/dexined_biped.pth]

Requires: gdown  (pip install gdown)
"""

import argparse
import os
import sys


GDRIVE_FILE_ID = "1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "dexined_biped.pth"
)


def main():
    parser = argparse.ArgumentParser(description="Download DexiNed BIPED weights")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        print(f"Weights already exist at {out_path}")
        return

    try:
        import gdown
    except ImportError:
        print("ERROR: gdown is required. Install with: pip install gdown")
        print(f"\nAlternatively, download manually from:")
        print(f"  https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view?usp=sharing")
        print(f"  and save to: {out_path}")
        sys.exit(1)

    print(f"Downloading DexiNed BIPED weights to {out_path} ...")
    gdown.download(GDRIVE_URL, out_path, quiet=False)

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"Done. File size: {size_mb:.1f} MB")
    else:
        print("ERROR: Download failed.")
        print(f"Download manually from:")
        print(f"  https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view?usp=sharing")
        print(f"  and save to: {out_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
