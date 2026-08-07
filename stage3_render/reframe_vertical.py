"""
Stage 3a: Vertical reframing using OpenCV + mediapipe face detection.

Tracks the speaker's face across the clip and produces a smoothly-panned
9:16 crop, instead of a static center crop that can cut off the speaker
when they move.

Approach:
    1. Sample face bounding boxes across the clip using mediapipe.
    2. Smooth the face-center x-position over time (exponential moving average)
       so the crop doesn't jitter frame-to-frame.
    3. Re-render frames cropped to 9:16 around the smoothed center.
    4. Mux the original audio back on with ffmpeg (OpenCV can't write audio).

Usage:
    python reframe_vertical.py --input clip_001.mp4 --output clip_001_vertical.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection


def detect_face_centers(video_path: str, sample_every_n_frames: int = 2):
    """
    Returns a list of (frame_index, center_x_norm) for frames where a face
    was detected. center_x_norm is 0-1 (fraction of frame width).
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    centers = []

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                if results.detections:
                    best = max(
                        results.detections,
                        key=lambda d: d.location_data.relative_bounding_box.width
                        * d.location_data.relative_bounding_box.height,
                    )
                    bbox = best.location_data.relative_bounding_box
                    center_x = bbox.xmin + bbox.width / 2
                    centers.append((frame_idx, center_x))

            frame_idx += 1

    cap.release()
    return centers, frame_idx


def build_smoothed_center_path(centers, total_frames: int, smoothing_alpha: float = 0.08):
    """
    Fills in a center_x value for every frame (carrying forward the last known
    detection when no face was found), then applies an exponential moving
    average so the crop pans smoothly instead of jumping.
    """
    if not centers:
        return [0.5] * total_frames

    center_by_frame = {}
    for frame_idx, cx in centers:
        center_by_frame[frame_idx] = cx

    raw_path = []
    last_known = centers[0][1]
    for i in range(total_frames):
        if i in center_by_frame:
            last_known = center_by_frame[i]
        raw_path.append(last_known)

    smoothed = [raw_path[0]]
    for cx in raw_path[1:]:
        smoothed.append(smoothing_alpha * cx + (1 - smoothing_alpha) * smoothed[-1])

    return smoothed


def render_cropped_video(video_path: str, output_path: str, center_path, target_aspect: float = 9 / 16):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_h = src_h
    crop_w = int(crop_h * target_aspect)
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(crop_w / target_aspect)

    out_w, out_h = 1080, 1920
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        center_x_norm = center_path[frame_idx] if frame_idx < len(center_path) else 0.5
        center_x_px = int(center_x_norm * src_w)

        x0 = max(0, min(src_w - crop_w, center_x_px - crop_w // 2))
        y0 = max(0, (src_h - crop_h) // 2)

        cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(resized)

        frame_idx += 1

    cap.release()
    writer.release()


def mux_audio(silent_video_path: str, source_video_path: str, final_output_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", silent_video_path,
        "-i", source_video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        final_output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] ffmpeg mux failed:")
        print(result.stderr[-2000:])
        sys.exit(1)


def reframe_video(input_path: str, output_path: str, sample_every_n_frames: int = 2, smoothing_alpha: float = 0.08):
    print(f"[INFO] Detecting face positions in {input_path} ...")
    centers, total_frames = detect_face_centers(input_path, sample_every_n_frames)
    print(f"[INFO] Found face in {len(centers)}/{total_frames} sampled frame checks")

    center_path = build_smoothed_center_path(centers, total_frames, smoothing_alpha)

    tmp_silent_path = str(Path(output_path).with_suffix(".silent.mp4"))
    print(f"[INFO] Rendering cropped frames -> {tmp_silent_path}")
    render_cropped_video(input_path, tmp_silent_path, center_path)

    print(f"[INFO] Muxing audio -> {output_path}")
    mux_audio(tmp_silent_path, input_path, output_path)

    Path(tmp_silent_path).unlink(missing_ok=True)
    print(f"[INFO] Done: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reframe a video to 9:16 with face tracking.")
    parser.add_argument("--input", required=True, help="Path to source clip (already cut to the target segment)")
    parser.add_argument("--output", required=True, help="Path to save the vertical reframed clip")
    parser.add_argument("--sample-every-n-frames", type=int, default=2)
    parser.add_argument("--smoothing-alpha", type=float, default=0.08, help="Lower = smoother/slower pan, higher = snappier tracking")
    args = parser.parse_args()

    reframe_video(args.input, args.output, args.sample_every_n_frames, args.smoothing_alpha)