"""
Multi-Video Single-Object Real-Time Tracker
==========================================

Tracks one moving object per video across a batch of videos using OpenCV trackers
(CSRT/KCF/MOSSE) with optional auto-initialization and Kalman smoothing.

Outputs per-video:
- Annotated MP4 with bounding box and trajectory.
- CSV with frame index, timestamp (s), bbox [x, y, w, h], centroid [cx, cy].

Usage examples
--------------
# Track all videos in a folder; select ROI once per video on the first frame
python multi_video_tracker.py --videos ./videos --output ./runs/track --tracker csrt --display

# Auto-initialize ROI via motion on the first 60 frames (no manual click)
python multi_video_tracker.py --videos ./videos --output ./runs/track --auto-init --auto-frames 60

# Track specific files
python multi_video_tracker.py --videos vid1.mp4 vid2.mp4 --output ./runs/track --tracker kcf --no-display

# Faster preview by resizing to width=960 (keeps aspect ratio)
python multi_video_tracker.py --videos ./videos --output ./runs/track --max-width 960

Requirements
------------
- Python 3.8+
- OpenCV-Python (cv2) >= 4.5

Install: pip install opencv-python
(Optional for speed): pip install opencv-contrib-python (for extra trackers if needed)
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Utility: Simple 2D Kalman Filter for centroid smoothing
# -----------------------------
class Kalman2D:
    """Constant-velocity Kalman filter for 2D points.

    State vector: [x, y, vx, vy]^T
    Measurement: [x, y]
    """

    def __init__(self, dt: float = 1.0, process_var: float = 1e-2, meas_var: float = 5.0):
        self.dt = dt
        # State transition matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        # Process noise covariance
        q = process_var
        self.Q = q * np.array([[dt**4/4, 0, dt**3/2, 0],
                               [0, dt**4/4, 0, dt**3/2],
                               [dt**3/2, 0, dt**2, 0],
                               [0, dt**3/2, 0, dt**2]], dtype=np.float32)
        # Measurement noise covariance
        r = meas_var
        self.R = r * np.eye(2, dtype=np.float32)
        # State and covariance initialization
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32)
        self.initialized = False

    def init(self, x: float, y: float):
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, zx: float, zy: float):
        z = np.array([[zx], [zy]], dtype=np.float32)
        y = z - (self.H @ self.x)  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])


# -----------------------------
# Tracker factory
# -----------------------------

def create_tracker(name: str):
    name = name.lower()
    if name == "csrt":
        return cv2.TrackerCSRT_create()
    if name == "kcf":
        return cv2.TrackerKCF_create()
    if name == "mosse":
        return cv2.legacy.TrackerMOSSE_create() if hasattr(cv2, "legacy") else cv2.TrackerMOSSE_create()
    raise ValueError(f"Unknown tracker: {name}. Choose from [csrt, kcf, mosse]")


# -----------------------------
# Auto-initialize bbox via motion detection on first N frames
# -----------------------------

def auto_initialize_bbox(cap: cv2.VideoCapture, frames: int = 60, min_area: int = 500) -> Optional[Tuple[int, int, int, int]]:
    backsub = cv2.createBackgroundSubtractorMOG2(history=frames, varThreshold=25, detectShadows=False)
    accum = None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read and accumulate motion
    read_frames = 0
    positions = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while read_frames < frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = backsub.apply(gray)
        fg = cv2.medianBlur(fg, 5)
        if accum is None:
            accum = fg.astype(np.uint16)
        else:
            accum = np.clip(accum + fg.astype(np.uint16), 0, 65535)
        read_frames += 1

    # Restore position to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, positions)

    if accum is None:
        return None

    accum_u8 = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(accum_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest moving blob
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None

    x, y, bw, bh = cv2.boundingRect(c)
    # Clip bbox to frame
    x = max(0, x)
    y = max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    return (x, y, bw, bh)


# -----------------------------
# Video processing
# -----------------------------

def process_video(
    video_path: Path,
    out_dir: Path,
    tracker_name: str = "csrt",
    display: bool = False,
    write_video: bool = True,
    auto_init: bool = False,
    auto_frames: int = 60,
    max_width: Optional[int] = None,
    kalman: bool = True,
) -> Tuple[bool, str]:
    """
    Returns (ok, message). On success, writes outputs to out_dir.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, f"Could not open: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional resize scale
    scale = 1.0
    if max_width is not None and width > max_width:
        scale = max_width / float(width)
        width = int(width * scale)
        height = int(height * scale)

    # Read first frame
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return False, f"Empty/invalid video: {video_path}"

    if scale != 1.0:
        frame = cv2.resize(frame, (width, height))

    # Initialize bbox
    if auto_init:
        # Rewind to the beginning for auto init; we'll consume frames again afterward
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        bbox = auto_initialize_bbox(cap, frames=auto_frames)
        # After auto init, read the true first frame again to align streams
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return False, f"Failed to re-read first frame: {video_path}"
        if scale != 1.0:
            frame = cv2.resize(frame, (width, height))
        if bbox is None:
            return False, f"Auto-init failed to find motion in: {video_path}"
    else:
        # Manual ROI selection
        sel = frame.copy()
        title = f"Select ROI: {video_path.name} (press ENTER or SPACE to confirm, c to cancel)"
        bbox = cv2.selectROI(title, sel, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        if bbox is None or bbox == (0, 0, 0, 0):
            return False, f"ROI selection canceled/invalid for: {video_path}"

    tracker = create_tracker(tracker_name)
    tracker.init(frame, tuple(map(int, bbox)))

    # Prepare output writers
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / f"{stem}_tracked.mp4"), fourcc, fps, (width, height))

    csv_path = out_dir / f"{stem}_track.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "time_s", "x", "y", "w", "h", "cx", "cy"])  # header

    # Kalman filter for smoothing centroid
    kf = Kalman2D(dt=dt) if kalman else None
    if kf:
        x, y, w, h = map(float, bbox)
        kf.init(x + w / 2.0, y + h / 2.0)

    frame_idx = 0
    ok_track = True

    while True:
        if frame_idx > 0:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (width, height))

        ok, box = tracker.update(frame)
        if not ok:
            ok_track = False
            # Show lost message and try to continue
            cv2.putText(frame, "Tracking lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            box = None
        else:
            x, y, w, h = map(float, box)
            cx, cy = x + w / 2.0, y + h / 2.0
            if kf and kf.initialized:
                kf.predict()
                cx, cy = kf.update(cx, cy)
                # Adjust bbox to keep smoothed centroid, preserve size
                x, y = cx - w / 2.0, cy - h / 2.0

            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            # Trajectory point
            cv2.circle(frame, (int(cx), int(cy)), 3, (255, 0, 0), -1)

            # Write CSV row
            t = frame_idx / fps
            csv_writer.writerow([frame_idx, f"{t:.4f}", f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}", f"{cx:.1f}", f"{cy:.1f}"])

        # HUD
        cv2.putText(frame, f"Tracker: {tracker_name.upper()}  FPS: {fps:.1f}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if write_video and writer is not None:
            writer.write(frame)

        if display:
            cv2.imshow("Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        frame_idx += 1

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    csv_file.close()
    if display:
        cv2.destroyAllWindows()

    status_msg = f"Done: {video_path.name} -> {out_dir}"
    if not ok_track:
        status_msg += " (warning: tracking lost at some point)"
    return True, status_msg


# -----------------------------
# Batch runner
# -----------------------------

def gather_videos(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v", "*.webm"):
                files.extend(sorted(p.glob(ext)))
        elif p.is_file():
            files.append(p)
        else:
            print(f"Warning: not found {inp}")
    # De-dup and keep order
    seen = set()
    unique = []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique.append(f)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Batch single-object tracker for multiple videos")
    parser.add_argument("--videos", nargs="+", required=True,
                        help="One or more video files and/or folders containing videos")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for annotated videos and CSVs")
    parser.add_argument("--tracker", type=str, default="csrt", choices=["csrt", "kcf", "mosse"],
                        help="OpenCV tracker to use (default: csrt)")
    parser.add_argument("--auto-init", action="store_true",
                        help="Auto-initialize ROI using motion detection on the first N frames")
    parser.add_argument("--auto-frames", type=int, default=60,
                        help="Number of frames to analyze for auto-init (default: 60)")
    parser.add_argument("--no-display", dest="display", action="store_false",
                        help="Disable preview window for faster processing")
    parser.add_argument("--display", dest="display", action="store_true", default=False,
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-video", dest="write_video", action="store_false",
                        help="Do not save annotated video; only CSV")
    parser.add_argument("--max-width", type=int, default=None,
                        help="Resize frames to this width for speed (keeps aspect ratio)")
    parser.add_argument("--no-kalman", dest="kalman", action="store_false",
                        help="Disable Kalman smoothing of the centroid")

    args = parser.parse_args()

    out_dir = Path(args.output)
    videos = gather_videos(args.videos)
    if not videos:
        print("No videos found.")
        sys.exit(1)

    print(f"Found {len(videos)} videos")

    for v in videos:
        vid_out_dir = out_dir / v.stem
        ok, msg = process_video(
            video_path=v,
            out_dir=vid_out_dir,
            tracker_name=args.tracker,
            display=args.display,
            write_video=args.write_video,
            auto_init=args.auto_init,
            auto_frames=args.auto_frames,
            max_width=args.max_width,
            kalman=args.kalman,
        )
        print(("[OK]" if ok else "[ERR]"), msg)


if __name__ == "__main__":
    main()
