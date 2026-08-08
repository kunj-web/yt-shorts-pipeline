"""
Stage 1a: Batch video downloader using yt-dlp.

Reads a list of YouTube URLs from config/videos.txt and downloads each
at a capped resolution (1080p) into the working output directory.

Usage:
    python download_videos.py --input ../config/videos.txt --output /kaggle/working/raw --cookies /path/to/cookies.txt

Requires creator permission to be secured for every URL in videos.txt --
this script does not check licensing, that's on the operator.

Note: requires a JS runtime (deno) to be installed for YouTube's signature
challenge to solve correctly on headless/cloud environments like Kaggle.
Install with: curl -fsSL https://deno.land/install.sh | sh -s -- -y
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

YT_DLP_CMD = [sys.executable, "-m", "yt_dlp"]


def download_videos(input_file: str, output_dir: str, max_height: int = 1080, cookies_file: str = None):
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

    if cookies_file:
        cmd += ["--cookies", cookies_file]

    # Required for YouTube's JS signature challenge -- Kaggle/cloud environments
    # need this explicit remote component since no browser JS runtime is present.
    cmd += ["--remote-components", "ejs:github"]

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


def download_from_channel(channel_url: str, output_dir: str, latest_n: int = 10, max_height: int = 1080, cookies_file: str = None):
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

    if cookies_file:
        cmd += ["--cookies", cookies_file]

    cmd += ["--remote-components", "ejs:github"]

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
    parser.add_argument("--cookies", default=None, help="Path to a cookies.txt file exported from a logged-in browser session, to avoid YouTube bot-detection blocks on cloud IPs")
    args = parser.parse_args()

    if args.channel:
        download_from_channel(args.channel, args.output, args.latest_n, args.max_height, args.cookies)
    else:
        download_videos(args.input, args.output, args.max_height, args.cookies)