"""Generate a synthetic tennis serve video for testing.

Creates a short video simulating a behind-the-server view of a tennis court
with an animated ball that follows a realistic serve trajectory, including
the bounce in the service box.

Usage:
    python -m tennis_serve.generate_test_video [--output test_serve.mp4]
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np


# Video settings
WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION_S = 3.0  # total video length

# Court drawing parameters (pixel coordinates for a behind-the-server view)
# Vanishing-point perspective: lines converge toward the top-centre.
COURT_COLOR = (60, 120, 60)      # dark green surface
LINE_COLOR = (255, 255, 255)     # white lines
NET_COLOR = (200, 200, 200)

# Key court landmarks in pixel space (hand-tuned for a realistic perspective)
# Baseline (bottom of frame, closest to camera)
BL_LEFT = (280, 680)
BL_RIGHT = (1000, 680)

# Service line (mid-court)
SL_LEFT = (400, 400)
SL_RIGHT = (880, 400)

# Net
NET_LEFT = (440, 280)
NET_RIGHT = (840, 280)

# Centre service line meets the net and service line
CENTRE_BOTTOM = (640, 400)   # at service line
CENTRE_NET = (640, 280)      # at net

# Singles sidelines at various depths
# (we interpolate between baseline and net for perspective)

# Service box corners (deuce side = right half when looking from behind server)
SBOX_DEUCE = [
    CENTRE_NET,       # near-left  (centre line at net)
    NET_RIGHT,        # near-right (sideline at net)
    SL_RIGHT,         # far-right  (sideline at service line)
    CENTRE_BOTTOM,    # far-left   (centre line at service line)
]


def lerp(a, b, t):
    """Linear interpolation between two points."""
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t))


def draw_court(frame):
    """Draw a simplified tennis court from the behind-the-server perspective."""
    # Green surface
    frame[:] = COURT_COLOR

    # Darker area outside the court
    pts_outer = np.array([
        [0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT],
    ], dtype=np.int32)
    pts_court = np.array([
        list(lerp(BL_LEFT, NET_LEFT, -0.15)),
        list(lerp(BL_RIGHT, NET_RIGHT, -0.15)),
        list(lerp(BL_RIGHT, NET_RIGHT, 1.3)),
        list(lerp(BL_LEFT, NET_LEFT, 1.3)),
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts_outer], (40, 80, 40))
    cv2.fillPoly(frame, [pts_court], COURT_COLOR)

    # Baseline
    cv2.line(frame, BL_LEFT, BL_RIGHT, LINE_COLOR, 2)

    # Service line
    cv2.line(frame, SL_LEFT, SL_RIGHT, LINE_COLOR, 2)

    # Centre service line
    cv2.line(frame, CENTRE_NET, CENTRE_BOTTOM, LINE_COLOR, 2)

    # Singles sidelines
    cv2.line(frame, BL_LEFT, lerp(BL_LEFT, NET_LEFT, 1.15), LINE_COLOR, 2)
    cv2.line(frame, BL_RIGHT, lerp(BL_RIGHT, NET_RIGHT, 1.15), LINE_COLOR, 2)

    # Net
    cv2.line(frame, NET_LEFT, NET_RIGHT, NET_COLOR, 3)

    # Net posts
    cv2.circle(frame, NET_LEFT, 5, NET_COLOR, -1)
    cv2.circle(frame, NET_RIGHT, 5, NET_COLOR, -1)


def ball_trajectory(t, total_t):
    """Compute ball (x, y) pixel position and apparent radius at time t.

    Simulates a serve from bottom-centre flying toward the deuce service box,
    crossing the net, bouncing, and continuing.

    Parameters
    ----------
    t : float
        Time in seconds from the start of the serve motion.
    total_t : float
        Total flight time.

    Returns
    -------
    (x, y, radius, visible)
    """
    # Normalised time
    p = t / total_t

    if p < 0:
        return 640, 660, 0, False

    # In a behind-the-server view:
    #   - y DECREASES as the ball flies away from the camera
    #   - Near the bounce, y INCREASES briefly (ball drops toward court)
    #   - After the bounce, y DECREASES again (ball rises)
    #
    # Phase 1: flight from server toward service box (0 → 0.50)
    #   Sub-phase A (0 → 0.30): ball rises after racket contact, y decreases fast
    #   Sub-phase B (0.30 → 0.50): ball descends toward bounce, y INCREASES
    # Phase 2: after bounce (0.50 → 1.0): ball rises, y decreases again

    bounce_p = 0.50

    if p <= bounce_p:
        frac = p / bounce_p  # 0→1 over flight phase
        # X: drifts toward deuce side
        x = 640 + 110 * frac
        # Base Y: linear movement from 670 (near camera) toward ~420 (service box)
        base_y = 670 - 250 * frac
        # Arc: the ball goes UP (negative y offset) in the first half of flight,
        # then comes DOWN (positive y offset) as it drops to the bounce.
        # At frac=0 → arc=0, frac≈0.4 → arc=-40 (highest point), frac=1 → arc=+25
        arc = -55 * math.sin(frac * math.pi * 0.7) + 25 * frac
        y = base_y + arc
        radius = int(10 - 4 * frac)
    else:
        frac = (p - bounce_p) / (1.0 - bounce_p)  # 0→1 after bounce
        # Bounce point is at approximately y=445, x=750
        x = 750 + 50 * frac
        # After bounce: ball rises sharply (y decreases)
        y = 445 - 150 * frac
        # Slight arc on the rise
        arc = -20 * math.sin(frac * math.pi)
        y += arc
        radius = int(6 - 2 * frac)

    radius = max(3, radius)
    return int(x), int(y), radius, True


def generate(output_path: str):
    """Generate the synthetic test video."""
    total_frames = int(FPS * DURATION_S)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))

    # Ball appears after a short delay (simulating the toss)
    serve_start_frame = int(FPS * 0.3)
    serve_flight_time = DURATION_S - 0.3

    # Pre-render the court background
    bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    draw_court(bg)

    for i in range(total_frames):
        frame = bg.copy()
        t = (i - serve_start_frame) / FPS

        x, y, radius, visible = ball_trajectory(t, serve_flight_time)

        if visible and i >= serve_start_frame:
            # Ball shadow
            cv2.circle(frame, (x + 3, y + 3), radius, (30, 60, 30), -1)
            # Tennis ball (yellow-green)
            cv2.circle(frame, (x, y), radius, (0, 230, 230), -1)
            cv2.circle(frame, (x, y), radius, (0, 200, 200), 1)

        # HUD text
        cv2.putText(
            frame, "SYNTHETIC TEST — Tennis Serve Analyzer",
            (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )
        cv2.putText(
            frame, f"Frame {i}/{total_frames}",
            (WIDTH - 200, HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
        )

        writer.write(frame)

    writer.release()
    print(f"Generated test video: {output_path}")
    print(f"  {WIDTH}x{HEIGHT} @ {FPS} fps, {total_frames} frames, {DURATION_S}s")
    print(f"\nService box corners (deuce side) for calibration:")
    print(f"  --points '{list(SBOX_DEUCE)}'")


def main():
    p = argparse.ArgumentParser(description="Generate a synthetic tennis serve test video.")
    p.add_argument("--output", "-o", default="test_serve.mp4", help="Output path.")
    args = p.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
