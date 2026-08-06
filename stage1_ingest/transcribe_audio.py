"""
Stage 1b: Transcription using faster-whisper.

Produces word-level timestamped transcripts (JSON) from downloaded videos.

Usage:
    python transcribe_audio.py --input-dir ./test_output --output-dir ./test_transcripts --device cpu --compute-type int8
"""

import argparse
import json
import os
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_file(model: WhisperModel, video_path: str) -> dict:
    segments, info = model.transcribe(video_path, word_timestamps=True, vad_filter=True)

    result = {
        "source_file": os.path.basename(video_path),
        "language": info.language,
        "duration": info.duration,
        "segments": [],
    }

    for seg in segments:
        segment_data = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in (seg.words or [])
            ],
        }
        result["segments"].append(segment_data)

    return result


def transcribe_directory(input_dir: str, output_dir: str, model_size="medium", device="cuda", compute_type="float16"):
    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted(Path(input_dir).glob("*.mp4"))
    if not video_files:
        print(f"[WARN] No .mp4 files found in {input_dir}")
        return []

    print(f"[INFO] Loading faster-whisper model '{model_size}' on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    output_paths = []
    for video_path in video_files:
        video_id = video_path.stem
        out_path = Path(output_dir) / f"{video_id}.json"

        if out_path.exists():
            print(f"[SKIP] Transcript already exists for {video_id}")
            output_paths.append(str(out_path))
            continue

        print(f"[INFO] Transcribing {video_path.name} ...")
        try:
            transcript = transcribe_file(model, str(video_path))
        except Exception as e:
            print(f"[ERROR] Failed to transcribe {video_path.name}: {e}")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved transcript -> {out_path}")
        output_paths.append(str(out_path))

    print(f"[INFO] Completed {len(output_paths)} transcript(s).")
    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe downloaded videos with faster-whisper.")
    parser.add_argument("--input-dir", default="/kaggle/working/raw")
    parser.add_argument("--output-dir", default="/kaggle/working/transcripts")
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()

    transcribe_directory(args.input_dir, args.output_dir, args.model_size, args.device, args.compute_type)