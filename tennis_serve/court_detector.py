"""Court line and service box detection for tennis serve analysis.

Detects the court lines visible from a behind-the-server camera angle using
edge detection and Hough line transforms, then identifies the service box
region for in/out determination.

Because automatic detection is unreliable across different broadcast feeds,
this module also supports **manual calibration** where the user clicks four
corners of the service box in the first frame.
"""

import cv2
import numpy as np


# Standard tennis court dimensions in metres (ITF rules).
COURT_WIDTH = 10.97  # doubles sideline to sideline
SINGLES_WIDTH = 8.23  # singles sideline to sideline
SERVICE_BOX_DEPTH = 6.40  # from net to service line
HALF_COURT_WIDTH = SINGLES_WIDTH / 2  # centre line splits service boxes
NET_TO_BASELINE = 11.89  # net to baseline


class CourtCalibration:
    """Perspective mapping between image pixels and real-world court coords.

    The user (or auto-detection) provides four image points corresponding to
    the four corners of the *target service box* (the one the serve must land
    in).  We compute a homography to map pixel positions to real-world metre
    coordinates on the court surface.

    Real-world coordinate system (metres, origin at net-centre):
        - X axis: along the net (positive = deuce side)
        - Y axis: perpendicular to the net (positive = towards the far baseline)
    """

    def __init__(self):
        self.homography = None
        self.inv_homography = None
        self.image_points = None
        # Real-world corners of a service box (deuce side by default).
        # Order: top-left, top-right, bottom-right, bottom-left as seen from
        # behind the server (i.e. closest to camera first).
        self.world_points = None

    def calibrate_deuce(self, image_points):
        """Calibrate for the deuce (right-side) service box.

        image_points: list of 4 (x,y) pixel coords in order:
            0 - near-left  (centre service line at net)
            1 - near-right (singles sideline at net)
            2 - far-right  (singles sideline at service line)
            3 - far-left   (centre service line at service line)
        """
        self.world_points = np.array([
            [0.0, 0.0],                          # centre mark at net
            [HALF_COURT_WIDTH, 0.0],              # right sideline at net
            [HALF_COURT_WIDTH, SERVICE_BOX_DEPTH],  # right sideline at service line
            [0.0, SERVICE_BOX_DEPTH],             # centre at service line
        ], dtype=np.float32)
        self._compute(image_points)

    def calibrate_ad(self, image_points):
        """Calibrate for the ad (left-side) service box."""
        self.world_points = np.array([
            [-HALF_COURT_WIDTH, 0.0],
            [0.0, 0.0],
            [0.0, SERVICE_BOX_DEPTH],
            [-HALF_COURT_WIDTH, SERVICE_BOX_DEPTH],
        ], dtype=np.float32)
        self._compute(image_points)

    def _compute(self, image_points):
        self.image_points = np.array(image_points, dtype=np.float32)
        self.homography, _ = cv2.findHomography(
            self.image_points, self.world_points,
        )
        self.inv_homography, _ = cv2.findHomography(
            self.world_points, self.image_points,
        )

    @property
    def is_calibrated(self):
        return self.homography is not None

    def pixel_to_court(self, px, py):
        """Convert a pixel position to real-world court coordinates (metres)."""
        if not self.is_calibrated:
            return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.homography)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def court_to_pixel(self, cx, cy):
        """Convert court coordinates back to pixel position."""
        if not self.is_calibrated:
            return None
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.inv_homography)
        return int(out[0, 0, 0]), int(out[0, 0, 1])

    def is_in_service_box(self, court_x, court_y):
        """Return True if the court coordinate is inside the service box."""
        if self.world_points is None:
            return False
        x_min = min(self.world_points[:, 0])
        x_max = max(self.world_points[:, 0])
        y_min = min(self.world_points[:, 1])
        y_max = max(self.world_points[:, 1])
        return x_min <= court_x <= x_max and y_min <= court_y <= y_max


class CourtLineDetector:
    """Automatic court line detection using edge detection and Hough transform.

    This works best on clean broadcast footage with visible white court lines.
    For noisy or low-contrast footage, manual calibration is recommended.
    """

    def __init__(self, canny_low=50, canny_high=150, hough_threshold=100):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold

    def detect_lines(self, frame):
        """Detect straight lines in the frame.

        Returns a list of (rho, theta) pairs from the Hough transform, and
        a list of (x1, y1, x2, y2) line segments.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # White line isolation: court lines are typically bright white
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=white_mask)

        edges = cv2.Canny(gray_masked, self.canny_low, self.canny_high)

        # Dilate to connect broken edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=50,
            maxLineGap=20,
        )

        segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                segments.append((x1, y1, x2, y2))

        return segments, edges

    def classify_lines(self, segments):
        """Split detected segments into roughly horizontal and vertical groups."""
        horizontal = []
        vertical = []
        for x1, y1, x2, y2 in segments:
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < 30:
                horizontal.append((x1, y1, x2, y2))
            elif angle > 60:
                vertical.append((x1, y1, x2, y2))
        return horizontal, vertical


def manual_calibration(frame, service_box_side="deuce"):
    """Open an interactive window for the user to click the 4 service box corners.

    Parameters
    ----------
    frame : np.ndarray
        The first frame of the video.
    service_box_side : str
        ``"deuce"`` or ``"ad"`` — which service box the serve targets.

    Returns
    -------
    CourtCalibration
        A calibrated mapping object.
    """
    points = []
    labels = [
        "near-left (centre line at net)",
        "near-right (sideline at net)",
        "far-right (sideline at service line)",
        "far-left (centre line at service line)",
    ]

    display = frame.copy()
    window_name = "Click 4 service box corners (press 'r' to reset, 'q' to quit)"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            idx = len(points) - 1
            cv2.putText(
                display, f"{idx}: {labels[idx]}",
                (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1,
            )
            cv2.imshow(window_name, display)

    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, on_mouse)

    # Draw instructions
    for i, label in enumerate(labels):
        cv2.putText(
            display, f"Point {i}: {label}",
            (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 0), 1,
        )
    cv2.imshow(window_name, display)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("r"):
            points.clear()
            display[:] = frame
            for i, label in enumerate(labels):
                cv2.putText(
                    display, f"Point {i}: {label}",
                    (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 0), 1,
                )
            cv2.imshow(window_name, display)
        if key == ord("q"):
            break
        if len(points) == 4:
            break

    cv2.destroyWindow(window_name)

    if len(points) != 4:
        raise RuntimeError("Calibration cancelled — need exactly 4 points.")

    cal = CourtCalibration()
    if service_box_side == "deuce":
        cal.calibrate_deuce(points)
    else:
        cal.calibrate_ad(points)
    return cal
