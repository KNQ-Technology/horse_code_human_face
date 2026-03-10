from __future__ import annotations

from dataclasses import dataclass

from horse_id.config import FusionConfig


@dataclass
class TrackStatePayload:
    status: str
    stable_id: str
    bad_frame_count: int
    lost_frame_count: int
    hold_left_frames: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "status": self.status,
            "stable_id": self.stable_id,
            "bad_frame_count": self.bad_frame_count,
            "lost_frame_count": self.lost_frame_count,
            "hold_left_frames": self.hold_left_frames,
        }


@dataclass
class _TrackState:
    status: str = "UNCONFIRMED"
    stable_id: str = ""
    bad_frame_count: int = 0
    lost_frame_count: int = 0
    hold_left_frames: int = 0


class TrackStateMachine:
    """Manage per-track confirmation state with hold/lost control."""

    def __init__(self, config: FusionConfig, fps: float) -> None:
        self.config = config
        self._states: dict[int, _TrackState] = {}
        hold_frames = int(round(config.hold_seconds * fps))
        self._hold_frames_default = max(1, hold_frames)

    def begin_frame(self) -> None:
        for state in self._states.values():
            state.lost_frame_count += 1

    def update(self, track_id: int | None, fused_ready: bool, fused_id: str) -> TrackStatePayload:
        if track_id is None:
            return TrackStatePayload(
                status="UNCONFIRMED",
                stable_id="",
                bad_frame_count=0,
                lost_frame_count=0,
                hold_left_frames=0,
            )

        state = self._states.setdefault(track_id, _TrackState())
        state.lost_frame_count = 0

        if state.status == "UNCONFIRMED":
            if fused_ready and fused_id:
                state.status = "CONFIRMED"
                state.stable_id = fused_id
                state.bad_frame_count = 0
                state.hold_left_frames = self._hold_frames_default
        elif state.status == "CONFIRMED":
            if fused_ready and fused_id == state.stable_id:
                state.bad_frame_count = 0
                state.hold_left_frames = self._hold_frames_default
            elif fused_ready and fused_id:
                state.bad_frame_count += 1
            else:
                state.bad_frame_count += 1

            if state.bad_frame_count >= self.config.bad_frame_trigger:
                state.status = "HOLD"
                state.bad_frame_count = 0
                state.hold_left_frames = self._hold_frames_default
        else:
            if fused_ready and fused_id:
                state.status = "CONFIRMED"
                state.stable_id = fused_id
                state.bad_frame_count = 0
                state.hold_left_frames = self._hold_frames_default
            else:
                state.hold_left_frames = max(0, state.hold_left_frames - 1)
                if state.hold_left_frames == 0:
                    if state.stable_id:
                        state.status = "CONFIRMED"
                        state.hold_left_frames = self._hold_frames_default
                    else:
                        state.status = "UNCONFIRMED"
                        state.bad_frame_count = 0

        return TrackStatePayload(
            status=state.status,
            stable_id=state.stable_id,
            bad_frame_count=state.bad_frame_count,
            lost_frame_count=state.lost_frame_count,
            hold_left_frames=state.hold_left_frames,
        )

    def end_frame(self) -> None:
        stale_ids = [
            track_id
            for track_id, state in self._states.items()
            if state.lost_frame_count >= self.config.lost_frame_limit
        ]
        for track_id in stale_ids:
            self._states.pop(track_id, None)
