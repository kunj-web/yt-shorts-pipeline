"""
Stage 2: Viral clip selection using a local LLM (Llama 3.1/3.2 GGUF via llama-cpp-python).

Reads word-level transcripts produced by Stage 1 and asks the LLM to identify
self-contained, hook-y 30-90 second segments worth turning into short clips.

Runs on Kaggle's GPU via llama-cpp-python (n_gpu_layers=-1) since Ollama's
background daemon model doesn't work inside Kaggle's sandboxed notebook env.

Usage (on Kaggle):
    python select_viral_clips.py \
        --transcripts-dir /kaggle/working/transcripts \
        --output-dir /kaggle/working/clip_picks \
        --model-path /kaggle/working/models/llama-3.1-8b-instruct.Q4_K_M.gguf
"""

import argparse
import json
import os
import re
from pathlib import Path

from llama_cpp import Llama


DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "clip_selection_prompt.txt")


def load_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def build_transcript_text(transcript: dict) -> str:
    """
    Flattens the transcript into a simple timestamped text block the LLM can reason over.
    Keeps it compact -- segment-level, not word-level, to save context tokens.
    """
    lines = []
    for seg in transcript["segments"]:
        start = round(seg["start"], 1)
        end = round(seg["end"], 1)
        lines.append(f"[{start}s - {end}s] {seg['text']}")
    return "\n".join(lines)


def extract_json_array(raw_text: str):
    """
    LLMs sometimes wrap JSON in prose or code fences despite instructions.
    Pulls out the first [...] array found in the response.
    """
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in model output:\n{raw_text[:500]}")
    return json.loads(match.group(0))


def select_clips_for_transcript(
    llm: Llama,
    transcript: dict,
    prompt_template: str,
    min_seconds: int,
    max_seconds: int,
    clips_per_video: int,
) -> list:
    transcript_text = build_transcript_text(transcript)

    prompt = prompt_template.format(
        transcript_text=transcript_text,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        clips_per_video=clips_per_video,
    )

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.4,
    )

    raw_text = response["choices"][0]["message"]["content"]

    try:
        clips = extract_json_array(raw_text)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to parse LLM output: {e}")
        return []

    valid_clips = []
    for c in clips:
        if not all(k in c for k in ("start", "end", "reason")):
            continue
        duration = c["end"] - c["start"]
        if duration < min_seconds - 5 or duration > max_seconds + 15:
            continue
        valid_clips.append(c)

    return valid_clips


def process_directory(
    transcripts_dir: str,
    output_dir: str,
    model_path: str,
    prompt_path: str = DEFAULT_PROMPT_PATH,
    min_seconds: int = 30,
    max_seconds: int = 90,
    clips_per_video: int = 3,
    n_ctx: int = 16384,
    n_gpu_layers: int = -1,
):
    os.makedirs(output_dir, exist_ok=True)

    prompt_template = load_prompt_template(prompt_path)

    print(f"[INFO] Loading model from {model_path} (n_gpu_layers={n_gpu_layers})")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
        chat_format="llama-3",
    )

    transcript_files = sorted(Path(transcripts_dir).glob("*.json"))
    if not transcript_files:
        print(f"[WARN] No transcript files found in {transcripts_dir}")
        return []

    all_picks = []
    for t_path in transcript_files:
        video_id = t_path.stem
        out_path = Path(output_dir) / f"{video_id}.clips.json"

        if out_path.exists():
            print(f"[SKIP] Clip picks already exist for {video_id}")
            continue

        print(f"[INFO] Selecting clips for {video_id} ...")
        with open(t_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        clips = select_clips_for_transcript(
            llm, transcript, prompt_template, min_seconds, max_seconds, clips_per_video
        )

        if not clips:
            print(f"[WARN] No valid clips found for {video_id}")
            continue

        result = {"video_id": video_id, "source_file": transcript.get("source_file"), "clips": clips}

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved {len(clips)} clip pick(s) -> {out_path}")
        all_picks.append(result)

    print(f"[INFO] Completed clip selection for {len(all_picks)} video(s).")
    return all_picks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select viral clip candidates from transcripts using a local LLM.")
    parser.add_argument("--transcripts-dir", default="/kaggle/working/transcripts")
    parser.add_argument("--output-dir", default="/kaggle/working/clip_picks")
    parser.add_argument("--model-path", required=True, help="Path to a GGUF model file")
    parser.add_argument("--prompt-path", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--min-seconds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=90)
    parser.add_argument("--clips-per-video", type=int, default=3)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = offload all layers to GPU")
    args = parser.parse_args()

    process_directory(
        transcripts_dir=args.transcripts_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        prompt_path=args.prompt_path,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        clips_per_video=args.clips_per_video,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )