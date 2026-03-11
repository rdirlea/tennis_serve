"""Serve analysis: in/out classification, impact point, and speed estimation.

Works with the ball trajectory produced by ``BallDetector`` and the court
mapping from ``CourtCalibration`` to derive:

* **Impact point** — where the ball first contacts the court surface after
  crossing the net.  Detected by finding a sharp downward-to-upward change
  in vertical ball movement (the bounce).
* **In / Out** — whether the impact point falls within the target service box.
* **Serve speed** — estimated from the ball's displacement in world
  coordinates over the frames between the serve contact and the bounce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .court_detector import CourtCalibration


@dataclass
class ServeResult:
    """Container for the results of analysing a single serve."""

    is_in: bool | None = None
    impact_pixel: tuple[int, int] | None = None
    impact_court: tuple[float, float] | None = None
    speed_kmh: float | None = None
    trajectory_pixels: list[tuple[int, int]] = field(default_factory=list)
    trajectory_court: list[tuple[float, float]] = field(default_factory=list)
    bounce_frame_index: int | None = None


class ServeAnalyzer:
    """Analyse a tennis serve from tracked ball positions.

    Parameters
    ----------
    calibration : CourtCalibration
        A calibrated court mapping (pixel ↔ real-world metres).
    fps : float
        Video frame rate — needed to convert per-frame displacement to km/h.
    """

    def __init__(self, calibration: CourtCalibration, fps: float = 30.0):
        self.calibration = calibration
        self.fps = fps

    # ------------------------------------------------------------------
    # Bounce / impact detection
    # ------------------------------------------------------------------

    @staticmethod
    def find_bounce(positions: list[tuple[int, int]], window: int = 3):
        """Find the frame index where the ball bounces.

        The bounce is characterised by the ball moving *downward* in the image
        (increasing y) and then reversing to move *upward* (decreasing y) —
        because the camera is behind the server, the ball's y-coordinate first
        increases as it travels away and drops, then decreases after the bounce.

        We also look for a sudden change in the vertical velocity component
        using a short sliding window.

        Returns the index into *positions*, or ``None``.
        """
        if len(positions) < 2 * window + 1:
            return None

        # Compute smoothed vertical velocity (dy/dt).
        ys = np.array([p[1] for p in positions], dtype=np.float64)
        velocities = np.diff(ys)

        # Look for sign change: positive (going down in image) → negative
        # In a behind-the-server view, the ball going *away* from the camera
        # moves upward in the frame (y decreases) until it drops toward the
        # service box (y increases), then bounces back up (y decreases again).
        # So bounce = local *maximum* in y, i.e. vy goes from >0 to <0.
        best_idx = None
        best_change = 0.0

        for i in range(window, len(velocities) - window):
            before = np.mean(velocities[i - window:i])
            after = np.mean(velocities[i:i + window])
            # We want before > 0 (ball moving down / y increasing) and
            # after < 0 (ball moving up / y decreasing after bounce).
            change = before - after
            if before > 0 and after < 0 and change > best_change:
                best_change = change
                best_idx = i

        return best_idx

    # ------------------------------------------------------------------
    # Speed estimation
    # ------------------------------------------------------------------

    def estimate_speed(
        self,
        positions: list[tuple[int, int]],
        bounce_idx: int,
    ) -> float | None:
        """Estimate serve speed in km/h using court-mapped displacement.

        We measure the displacement between the first tracked position and the
        bounce point in real-world metres, divided by the elapsed time.

        This gives an *average* speed over the flight, which is a reasonable
        proxy for serve speed (comparable to broadcast radar-gun readings
        that measure speed just after the racket contact).
        """
        if not self.calibration.is_calibrated:
            return None

        if bounce_idx is None or bounce_idx < 1:
            return None

        # Use a point early in the trajectory (shortly after the serve contact)
        start_idx = max(0, bounce_idx // 4)
        start_px = positions[start_idx]
        end_px = positions[bounce_idx]

        start_court = self.calibration.pixel_to_court(*start_px)
        end_court = self.calibration.pixel_to_court(*end_px)

        if start_court is None or end_court is None:
            return None

        dx = end_court[0] - start_court[0]
        dy = end_court[1] - start_court[1]
        distance_m = math.hypot(dx, dy)

        # The ball also travels vertically (height). Approximate by adding
        # an assumed net-clearance arc of ~1.5 m.  This is rough but improves
        # the estimate compared to ignoring the vertical component entirely.
        arc_height_m = 1.5
        distance_m = math.hypot(distance_m, arc_height_m)

        frames_elapsed = bounce_idx - start_idx
        if frames_elapsed <= 0:
            return None

        time_s = frames_elapsed / self.fps
        speed_ms = distance_m / time_s
        speed_kmh = speed_ms * 3.6
        return speed_kmh

    # ------------------------------------------------------------------
    # Full serve analysis
    # ------------------------------------------------------------------

    def analyze(self, positions: list[tuple[int, int]]) -> ServeResult:
        """Run the full serve analysis pipeline on a trajectory.

        Parameters
        ----------
        positions : list of (x, y) pixel positions across frames.

        Returns
        -------
        ServeResult
        """
        result = ServeResult(trajectory_pixels=list(positions))

        # 1. Find the bounce point
        bounce_idx = self.find_bounce(positions)
        result.bounce_frame_index = bounce_idx

        if bounce_idx is not None:
            result.impact_pixel = positions[bounce_idx]

            # 2. Map to court coordinates
            if self.calibration.is_calibrated:
                court_pt = self.calibration.pixel_to_court(*result.impact_pixel)
                result.impact_court = court_pt

                # 3. In / out classification
                if court_pt is not None:
                    result.is_in = self.calibration.is_in_service_box(*court_pt)

                # Map full trajectory to court coords
                for px, py in positions:
                    c = self.calibration.pixel_to_court(px, py)
                    if c is not None:
                        result.trajectory_court.append(c)

            # 4. Speed estimation
            result.speed_kmh = self.estimate_speed(positions, bounce_idx)

        return result
