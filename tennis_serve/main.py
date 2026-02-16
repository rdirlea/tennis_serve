"""Main pipeline — CLI entry point for tennis serve analysis.

Usage
-----
    python -m tennis_serve.main VIDEO_PATH [options]

The pipeline:
1. Opens the video file.
2. Asks the user to calibrate the service box (click 4 corners) on the
   first frame — or accepts pre-set calibration points via ``--points``.
3. Processes every frame: detects/tracks the ball.
4. When the ball trajectory contains a bounce, runs serve analysis.
5. Displays the annotated video and optionally writes it to disk.
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2

from .ball_detector import BallDetector
from .court_detector import CourtCalibration, manual_calibration
from .serve_analyzer import ServeAnalyzer, ServeResult
from .visualizer import draw_frame


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Analyse tennis serve videos — detect in/out, impact point, and speed.",
    )
    p.add_argument("video", help="Path to the input video file (MP4/AVI).")
    p.add_argument(
        "--side", choices=["deuce", "ad"], default="deuce",
        help="Which service box the serve targets (default: deuce).",
    )
    p.add_argument(
        "--points",
        help=(
            "JSON string with 4 calibration points instead of interactive "
            "selection, e.g. '[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]'."
        ),
    )
    p.add_argument(
        "--output", "-o",
        help="Path to write the annotated output video. If omitted, only display.",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Do not open a display window (useful for headless processing).",
    )
    p.add_argument(
        "--hsv-lower", default="25,80,80",
        help="Lower HSV bound for ball detection (comma-separated, default: 25,80,80).",
    )
    p.add_argument(
        "--hsv-upper", default="65,255,255",
        help="Upper HSV bound for ball detection (comma-separated, default: 65,255,255).",
    )
    return p.parse_args(argv)


def run(argv: list[str] | None = None):
    args = parse_args(argv)

    # ── Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video '{args.video}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {args.video}")
    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Frames     : {total_frames}")

    # ── Calibration ───────────────────────────────────────────────────
    ret, first_frame = cap.read()
    if not ret:
        print("Error: cannot read the first frame.", file=sys.stderr)
        sys.exit(1)

    calibration = CourtCalibration()

    if args.points:
        pts = json.loads(args.points)
        if len(pts) != 4:
            print("Error: --points must contain exactly 4 [x,y] pairs.", file=sys.stderr)
            sys.exit(1)
        if args.side == "deuce":
            calibration.calibrate_deuce(pts)
        else:
            calibration.calibrate_ad(pts)
        print(f"Calibrated ({args.side} side) from command-line points.")
    else:
        if args.no_display:
            print(
                "Warning: no calibration points provided and --no-display is set. "
                "Serve analysis will be limited (no in/out or speed).",
                file=sys.stderr,
            )
        else:
            print("Please click the 4 service box corners in the window...")
            calibration = manual_calibration(first_frame, args.side)
            print("Calibration complete.")

    # ── Ball detector ─────────────────────────────────────────────────
    hsv_lo = tuple(int(v) for v in args.hsv_lower.split(","))
    hsv_hi = tuple(int(v) for v in args.hsv_upper.split(","))
    detector = BallDetector(hsv_lower=hsv_lo, hsv_upper=hsv_hi)

    # ── Serve analyzer ────────────────────────────────────────────────
    analyzer = ServeAnalyzer(calibration, fps=fps)

    # ── Video writer (optional) ───────────────────────────────────────
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # ── Process frames ────────────────────────────────────────────────
    # Re-wind to start (we already read the first frame for calibration).
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    result: ServeResult | None = None
    frame_idx = 0
    analysis_done = False

    print("\nProcessing... (press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ball_pos = detector.process_frame(frame)
        trajectory = detector.get_trajectory()

        # Run analysis once we have enough trajectory data and haven't yet.
        if not analysis_done and len(trajectory) > 20:
            result = analyzer.analyze(trajectory)
            if result.bounce_frame_index is not None:
                analysis_done = True
                _print_result(result)

        # Draw overlays
        draw_frame(frame, ball_pos, trajectory, calibration, result)

        # Frame counter
        cv2.putText(
            frame, f"Frame {frame_idx}/{total_frames}",
            (width - 220, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        if writer:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Tennis Serve Analysis", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                # Pause / resume on spacebar
                cv2.waitKey(0)

        frame_idx += 1

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"\nOutput written to {args.output}")
    if not args.no_display:
        cv2.destroyAllWindows()

    # Final analysis if not triggered during playback
    if not analysis_done and len(detector.get_trajectory()) > 10:
        result = analyzer.analyze(detector.get_trajectory())
        _print_result(result)

    print("\nDone.")


def _print_result(result: ServeResult):
    """Print serve analysis results to the console."""
    print("\n" + "=" * 45)
    print("  SERVE ANALYSIS RESULT")
    print("=" * 45)

    if result.is_in is not None:
        call = "IN" if result.is_in else "OUT"
        print(f"  Call          : {call}")
    else:
        print("  Call          : UNDETERMINED")

    if result.impact_pixel is not None:
        print(f"  Impact (px)   : {result.impact_pixel}")

    if result.impact_court is not None:
        cx, cy = result.impact_court
        print(f"  Impact (court): ({cx:.2f}, {cy:.2f}) metres")

    if result.speed_kmh is not None:
        print(f"  Speed         : {result.speed_kmh:.1f} km/h")
    else:
        print("  Speed         : N/A")

    if result.bounce_frame_index is not None:
        print(f"  Bounce frame  : {result.bounce_frame_index}")

    print("=" * 45 + "\n")


if __name__ == "__main__":
    run()
