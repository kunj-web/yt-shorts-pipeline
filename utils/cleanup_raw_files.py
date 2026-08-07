"""
Utility: deletes raw downloaded videos (and intermediate temp render files)
once a video's final clips have been exported -- keeps Kaggle's limited
session storage from filling up during a large batch run.

Usage:
    python cleanup_raw_files.py --raw-dir /kaggle/working/raw --output-dir /kaggle/working/clips --tmp-dir /kaggle/working/tmp_render
"""

import argparse
import shutil
from pathlib import Path


def cleanup(raw_dir: str, output_dir: str, tmp_dir: str, dry_run: bool = False):
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    if not raw_path.exists():
        print(f"[WARN] Raw dir not found: {raw_dir}")
        return

    exported_ids = {f.stem.split("_clip")[0] for f in output_path.glob("*.mp4")} if output_path.exists() else set()

    deleted = 0
    for video_file in raw_path.glob("*.mp4"):
        video_id = video_file.stem
        if video_id not in exported_ids:
            print(f"[SKIP] No exported clips found yet for {video_id}, keeping raw file")
            continue

        info_json = raw_path / f"{video_id}.info.json"

        if dry_run:
            print(f"[DRY-RUN] Would delete {video_file} and {info_json}")
            continue

        video_file.unlink(missing_ok=True)
        info_json.unlink(missing_ok=True)
        print(f"[INFO] Deleted raw source for {video_id}")
        deleted += 1

    tmp_path = Path(tmp_dir)
    if tmp_path.exists() and not dry_run:
        shutil.rmtree(tmp_path)
        print(f"[INFO] Cleared temp render directory: {tmp_dir}")

    print(f"[INFO] Cleanup complete. {deleted} raw video(s) deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete raw videos once their clips are exported.")
    parser.add_argument("--raw-dir", default="/kaggle/working/raw")
    parser.add_argument("--output-dir", default="/kaggle/working/clips")
    parser.add_argument("--tmp-dir", default="/kaggle/working/tmp_render")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be deleted without deleting")
    args = parser.parse_args()

    cleanup(args.raw_dir, args.output_dir, args.tmp_dir, args.dry_run)