"""
Stage 1a: Batch video downloader using yt-dlp.

Reads a list of YouTube URLs from config/videos.txt and downloads each
at a capped resolution (1080p) into the working output directory.

Usage:
    python download_videos.py --input ../config/videos.txt --output ./test_output
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Call yt-dlp as a Python module instead of relying on it being on PATH.
YT_DLP_CMD = [sys.executable, "-m", "yt_dlp"]


def download_videos(input_file: str, output_dir: str, max_height: int = 1080):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not urls:
        print("[WARN] No URLs found in input file. Nothing to download.")
        return []

    print(f"[INFO] Found {len(urls)} URL(s) to download.")

    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = YT_DLP_CMD + [
        "-a", input_file,
        "-f", f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--write-info-json",
    ]

    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] yt-dlp failed:")
        print(result.stderr[-2000:])
        sys.exit(1)

    print(result.stdout[-2000:])

    downloaded = sorted(Path(output_dir).glob("*.mp4"))
    print(f"[INFO] Downloaded {len(downloaded)} file(s) to {output_dir}")
    return [str(p) for p in downloaded]


def download_from_channel(channel_url: str, output_dir: str, latest_n: int = 10, max_height: int = 1080):
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = YT_DLP_CMD + [
        channel_url,
        "--playlist-end", str(latest_n),
        "-f", f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--write-info-json",
    ]

    print(f"[INFO] Pulling latest {latest_n} videos from {channel_url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] yt-dlp failed:")
        print(result.stderr[-2000:])
        sys.exit(1)

    downloaded = sorted(Path(output_dir).glob("*.mp4"))
    print(f"[INFO] Downloaded {len(downloaded)} file(s) to {output_dir}")
    return [str(p) for p in downloaded]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-download YouTube videos with yt-dlp.")
    parser.add_argument("--input", default="../config/videos.txt", help="Path to videos.txt")
    parser.add_argument("--output", default="/kaggle/working/raw", help="Directory to save downloads")
    parser.add_argument("--max-height", type=int, default=1080, help="Max video height (default 1080p)")
    parser.add_argument("--channel", default=None, help="Optional: channel/playlist URL instead of videos.txt")
    parser.add_argument("--latest-n", type=int, default=10, help="If --channel is used, how many latest videos to pull")
    args = parser.parse_args()

    if args.channel:
        download_from_channel(args.channel, args.output, args.latest_n, args.max_height)
    else:
        download_videos(args.input, args.output, args.max_height)