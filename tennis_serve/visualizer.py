"""Visualization overlays for tennis serve analysis.

Draws ball trajectory, impact point, service box outline, in/out label, and
speed estimate on video frames.
"""

import cv2
import numpy as np

from .court_detector import CourtCalibration
from .serve_analyzer import ServeResult


# Colours (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)


def draw_ball(frame, position, radius=6, color=YELLOW):
    """Draw the current ball position."""
    if position is not None:
        cv2.circle(frame, position, radius, color, -1)
        cv2.circle(frame, position, radius + 1, WHITE, 1)


def draw_trajectory(frame, positions, color=CYAN, thickness=2, max_points=60):
    """Draw the recent ball trajectory as a polyline with fading opacity."""
    pts = positions[-max_points:]
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, pts[i - 1], pts[i], c, thickness)


def draw_service_box(frame, calibration: CourtCalibration, color=WHITE, thickness=2):
    """Draw the calibrated service box outline on the frame."""
    if not calibration.is_calibrated or calibration.image_points is None:
        return
    pts = calibration.image_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def draw_impact(frame, result: ServeResult):
    """Draw the impact point with in/out label."""
    if result.impact_pixel is None:
        return

    color = GREEN if result.is_in else RED
    label = "IN" if result.is_in else "OUT"
    if result.is_in is None:
        color = ORANGE
        label = "?"

    px, py = result.impact_pixel

    # Cross-hair at impact
    size = 15
    cv2.line(frame, (px - size, py), (px + size, py), color, 2)
    cv2.line(frame, (px, py - size), (px, py + size), color, 2)
    cv2.circle(frame, (px, py), size, color, 2)

    # Label
    cv2.putText(
        frame, label, (px + size + 5, py + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
    )


def draw_hud(frame, result: ServeResult):
    """Draw a heads-up display with serve stats in the top-left corner."""
    y = 30
    line_h = 30

    def put(text, color=WHITE):
        nonlocal y
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += line_h

    put("SERVE ANALYSIS", YELLOW)

    if result.is_in is not None:
        color = GREEN if result.is_in else RED
        put(f"Call: {'IN' if result.is_in else 'OUT'}", color)
    else:
        put("Call: analyzing...", ORANGE)

    if result.speed_kmh is not None:
        put(f"Speed: {result.speed_kmh:.0f} km/h", CYAN)

    if result.impact_court is not None:
        cx, cy = result.impact_court
        put(f"Impact: ({cx:.2f}, {cy:.2f}) m", WHITE)


def draw_frame(
    frame,
    ball_pos,
    trajectory,
    calibration: CourtCalibration,
    result: ServeResult | None = None,
):
    """Composite all overlays onto a frame.

    Parameters
    ----------
    frame : np.ndarray
        The video frame (modified in-place).
    ball_pos : tuple or None
        Current ball position.
    trajectory : list of (x, y)
        Recent ball positions.
    calibration : CourtCalibration
        Court mapping (may be uncalibrated).
    result : ServeResult or None
        If available, draw impact point and stats.
    """
    draw_service_box(frame, calibration)
    draw_trajectory(frame, trajectory)
    draw_ball(frame, ball_pos)

    if result is not None:
        draw_impact(frame, result)
        draw_hud(frame, result)

    return frame
