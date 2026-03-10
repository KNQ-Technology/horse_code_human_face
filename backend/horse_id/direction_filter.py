"""Filter horses by horizontal movement direction using tracking data."""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DirectionFilter:
    """Classify each track as moving left-to-right or right-to-left.

    Uses a sliding window of bounding-box centre-x positions supplied by
    ByteTrack.  After *min_frames* observations the horizontal displacement
    determines direction.  Once decided the result is cached for the track's
    lifetime and never changes.

    Parameters
    ----------
    target_direction:
        ``"left_to_right"``, ``"right_to_left"``, or ``"both"``
        (disable filtering).
    min_frames:
        Minimum observations before committing a direction verdict.
    min_displacement_px:
        Minimum absolute horizontal displacement (in pixels) within the
        window to count as a definitive direction.  Prevents noise from
        nearly-stationary horses triggering a wrong classification.
    """

    def __init__(
        self,
        target_direction: str = "left_to_right",
        min_frames: int = 5,
        min_displacement_px: float = 30.0,
    ) -> None:
        self.target_direction = target_direction.lower().strip()
        self.min_frames = max(2, min_frames)
        self.min_displacement_px = min_displacement_px

        self._positions: dict[int, list[float]] = defaultdict(list)
        self._decided: dict[int, str] = {}

    def update(self, track_id: int | None, center_x: float) -> None:
        """Record a new centre-x observation for *track_id*."""
        if track_id is None:
            return
        self._positions[track_id].append(center_x)

        if track_id in self._decided:
            return

        pts = self._positions[track_id]
        if len(pts) < self.min_frames:
            return

        dx = pts[-1] - pts[0]
        if abs(dx) < self.min_displacement_px:
            return

        direction = "left_to_right" if dx > 0 else "right_to_left"
        self._decided[track_id] = direction
        logger.info(
            "direction decided: track=%s => %s (dx=%.1fpx over %d frames)",
            track_id, direction, dx, len(pts),
        )

    def should_process(self, track_id: int | None) -> bool:
        """Return True if the track should be processed (pass filter)."""
        if self.target_direction == "both":
            return True
        if track_id is None:
            return True
        direction = self._decided.get(track_id)
        if direction is None:
            return True
        return direction == self.target_direction
