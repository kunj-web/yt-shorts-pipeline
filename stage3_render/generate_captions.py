"""
Stage 3b: Caption generation -- builds an ASS subtitle file with word-by-word
sync, driven by Whisper's word-level timestamps from Stage 1.

Style: average-sized bold yellow text with a thick black outline for strong
contrast against any background footage.

Usage:
    python generate_captions.py \
        --transcript ../stage1_ingest/test_transcripts/VIDEOID.json \
        --clip-start 12.4 --clip-end 58.9 \
        --output clip_001.ass
"""

import argparse
import json


ASS_HEADER = """[Script Info]
Title: Auto-generated captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,68,&H0000FFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,5,0,2,60,60,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def seconds_to_ass_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def extract_words_in_range(transcript: dict, clip_start: float, clip_end: float):
    words = []
    for seg in transcript["segments"]:
        for w in seg.get("words", []):
            if w["end"] < clip_start or w["start"] > clip_end:
                continue
            rel_start = max(0.0, w["start"] - clip_start)
            rel_end = min(clip_end - clip_start, w["end"] - clip_start)
            if rel_end <= rel_start:
                continue
            words.append({"word": w["word"].strip(), "start": rel_start, "end": rel_end})
    return words


def build_ass_content(words: list) -> str:
    lines = [ASS_HEADER]
    for w in words:
        if not w["word"]:
            continue
        start_ts = seconds_to_ass_timestamp(w["start"])
        end_ts = seconds_to_ass_timestamp(w["end"])
        safe_word = w["word"].replace("{", "").replace("}", "")
        lines.append(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{safe_word}")
    return "\n".join(lines) + "\n"


def generate_captions(transcript_path: str, clip_start: float, clip_end: float, output_path: str):
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    words = extract_words_in_range(transcript, clip_start, clip_end)
    if not words:
        print(f"[WARN] No words found in range {clip_start}-{clip_end}s -- writing empty caption file")

    ass_content = build_ass_content(words)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    print(f"[INFO] Wrote {len(words)} word-events -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate word-by-word ASS captions for a clip.")
    parser.add_argument("--transcript", required=True, help="Path to Stage 1 transcript JSON")
    parser.add_argument("--clip-start", type=float, required=True, help="Clip start time in seconds (source video timeline)")
    parser.add_argument("--clip-end", type=float, required=True, help="Clip end time in seconds (source video timeline)")
    parser.add_argument("--output", required=True, help="Path to save the .ass file")
    args = parser.parse_args()

    generate_captions(args.transcript, args.clip_start, args.clip_end, args.output)