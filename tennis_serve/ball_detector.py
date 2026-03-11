"""Tennis ball detection and tracking using classical computer vision.

Uses HSV color filtering to isolate the tennis ball, contour analysis to
locate it in each frame, and a Kalman filter to maintain a smooth trajectory
even when detections are briefly lost.
"""

import cv2
import numpy as np


class KalmanTracker:
    """Kalman filter for 2D ball position and velocity tracking."""

    def __init__(self):
        # State: [x, y, vx, vy]  Measurement: [x, y]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32,
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        self.initialized = False
        self.frames_since_seen = 0
        self.max_frames_lost = 8

    def predict(self):
        if not self.initialized:
            return None
        state = self.kf.predict()
        return int(state[0, 0]), int(state[1, 0])

    def update(self, measurement):
        """Update with a detected (x, y) position."""
        meas = np.array([[np.float32(measurement[0])],
                         [np.float32(measurement[1])]])
        if not self.initialized:
            self.kf.statePre = np.array(
                [[meas[0, 0]], [meas[1, 0]], [0], [0]], dtype=np.float32,
            )
            self.kf.statePost = self.kf.statePre.copy()
            self.initialized = True
        self.kf.correct(meas)
        self.frames_since_seen = 0

    def mark_missed(self):
        self.frames_since_seen += 1

    @property
    def is_lost(self):
        return self.frames_since_seen > self.max_frames_lost

    def get_velocity(self):
        """Return current estimated velocity (vx, vy) in pixels/frame."""
        if not self.initialized:
            return 0.0, 0.0
        state = self.kf.statePost
        return float(state[2, 0]), float(state[3, 0])


class BallDetector:
    """Detect a tennis ball in a frame using HSV color segmentation.

    Parameters
    ----------
    hsv_lower : tuple
        Lower HSV bound for the tennis ball color (default: bright yellow-green).
    hsv_upper : tuple
        Upper HSV bound for the tennis ball color.
    min_radius : int
        Minimum ball radius in pixels to accept a detection.
    max_radius : int
        Maximum ball radius in pixels to accept a detection.
    """

    def __init__(
        self,
        hsv_lower=(25, 80, 80),
        hsv_upper=(65, 255, 255),
        min_radius=3,
        max_radius=30,
    ):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.tracker = KalmanTracker()

        # Store recent positions for trajectory analysis
        self.positions: list[tuple[int, int]] = []
        self.max_history = 300

    def detect(self, frame):
        """Detect the tennis ball in *frame*.

        Returns (x, y) centre of the ball or None if not found.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1

        predicted = self.tracker.predict()

        for cnt in contours:
            ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            # Circularity check
            area = cv2.contourArea(cnt)
            circle_area = np.pi * radius * radius
            if circle_area == 0:
                continue
            circularity = area / circle_area
            if circularity < 0.4:
                continue

            # Score: prefer candidates close to the predicted position
            score = circularity
            if predicted is not None:
                dist = np.hypot(cx - predicted[0], cy - predicted[1])
                # Penalise detections far from prediction
                score += max(0, 1.0 - dist / 200.0)

            if score > best_score:
                best_score = score
                best = (int(cx), int(cy))

        return best

    def process_frame(self, frame):
        """Run detection + tracking on a single frame.

        Returns the tracked (x, y) position or None.
        """
        detection = self.detect(frame)

        if detection is not None:
            self.tracker.update(detection)
            pos = detection
        else:
            self.tracker.mark_missed()
            pos = self.tracker.predict() if not self.tracker.is_lost else None

        if pos is not None:
            self.positions.append(pos)
            if len(self.positions) > self.max_history:
                self.positions.pop(0)

        return pos

    def get_trajectory(self):
        """Return the list of tracked ball positions."""
        return list(self.positions)

    def get_velocity(self):
        """Return current (vx, vy) in pixels/frame."""
        return self.tracker.get_velocity()

    def reset(self):
        """Clear tracking state for a new serve."""
        self.tracker = KalmanTracker()
        self.positions.clear()
