"""
Stage 3c: Final export -- ties together cutting, reframing, captioning, and
burning captions in, for every clip pick from Stage 2.

Usage:
    python export_clips.py \
        --raw-dir /kaggle/working/raw \
        --transcripts-dir /kaggle/working/transcripts \
        --clip-picks-dir /kaggle/working/clip_picks \
        --output-dir /kaggle/working/clips
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reframe_vertical import reframe_video
from generate_captions import generate_captions


def cut_segment(source_path: str, start: float, end: float, output_path: str):
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", source_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] ffmpeg cut failed:")
        print(result.stderr[-2000:])
        return False
    return True


def burn_captions(video_path: str, ass_path: str, output_path: str):
    escaped_ass_path = ass_path.replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"ass={escaped_ass_path}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] ffmpeg caption burn-in failed:")
        print(result.stderr[-2000:])
        return False
    return True


def export_clips_for_video(video_id, raw_dir, transcripts_dir, clip_picks_dir, output_dir, tmp_dir):
    source_path = Path(raw_dir) / f"{video_id}.mp4"
    transcript_path = Path(transcripts_dir) / f"{video_id}.json"
    picks_path = Path(clip_picks_dir) / f"{video_id}.clips.json"

    if not source_path.exists():
        print(f"[WARN] Source video not found for {video_id}, skipping")
        return []
    if not transcript_path.exists():
        print(f"[WARN] Transcript not found for {video_id}, skipping")
        return []
    if not picks_path.exists():
        print(f"[WARN] Clip picks not found for {video_id}, skipping")
        return []

    with open(picks_path, "r", encoding="utf-8") as f:
        picks = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    exported = []
    for i, clip in enumerate(picks["clips"]):
        start, end = clip["start"], clip["end"]
        clip_name = f"{video_id}_clip{i+1:02d}"
        print(f"[INFO] Exporting {clip_name} ({start}s - {end}s): {clip.get('hook', '')}")

        cut_path = str(Path(tmp_dir) / f"{clip_name}_cut.mp4")
        vertical_path = str(Path(tmp_dir) / f"{clip_name}_vertical.mp4")
        ass_path = str(Path(tmp_dir) / f"{clip_name}.ass")
        final_path = str(Path(output_dir) / f"{clip_name}.mp4")

        if not cut_segment(str(source_path), start, end, cut_path):
            continue

        try:
            reframe_video(cut_path, vertical_path)
        except Exception as e:
            print(f"[ERROR] Reframing failed for {clip_name}: {e}")
            continue

        generate_captions(str(transcript_path), start, end, ass_path)

        if not burn_captions(vertical_path, ass_path, final_path):
            continue

        print(f"[INFO] Saved final clip -> {final_path}")
        exported.append(final_path)

    return exported


def export_all(raw_dir, transcripts_dir, clip_picks_dir, output_dir, tmp_dir):
    pick_files = sorted(Path(clip_picks_dir).glob("*.clips.json"))
    if not pick_files:
        print(f"[WARN] No clip pick files found in {clip_picks_dir}")
        return []

    all_exported = []
    for pick_file in pick_files:
        video_id = pick_file.stem.replace(".clips", "")
        exported = export_clips_for_video(video_id, raw_dir, transcripts_dir, clip_picks_dir, output_dir, tmp_dir)
        all_exported.extend(exported)

    print(f"[INFO] Exported {len(all_exported)} clip(s) total.")
    return all_exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export final captioned vertical clips for all videos.")
    parser.add_argument("--raw-dir", default="/kaggle/working/raw")
    parser.add_argument("--transcripts-dir", default="/kaggle/working/transcripts")
    parser.add_argument("--clip-picks-dir", default="/kaggle/working/clip_picks")
    parser.add_argument("--output-dir", default="/kaggle/working/clips")
    parser.add_argument("--tmp-dir", default="/kaggle/working/tmp_render")
    args = parser.parse_args()

    export_all(args.raw_dir, args.transcripts_dir, args.clip_picks_dir, args.output_dir, args.tmp_dir)