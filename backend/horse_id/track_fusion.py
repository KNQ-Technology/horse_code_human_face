from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from horse_id.config import FusionConfig


@dataclass
class TrackVote:
    text: str
    conf: float
    valid: bool
    source: str = "ocr"


@dataclass
class FusedTrackResult:
    stable_id: str
    stable_conf: float
    vote_ratio: float
    support_count: int
    sample_count: int
    ready: bool

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "stable_id": self.stable_id,
            "stable_conf": self.stable_conf,
            "vote_ratio": self.vote_ratio,
            "support_count": self.support_count,
            "sample_count": self.sample_count,
            "ready": self.ready,
        }


class OCRTrackFuser:
    """Fuse frame-level OCR into a stable track-level ID."""

    VLM_WEIGHT_BOOST = 5.0
    LOCK_MIN_VOTES = 5
    LOCK_MIN_RATIO = 0.55

    def __init__(self, config: FusionConfig) -> None:
        self.config = config
        self._buffers: dict[int, deque[TrackVote]] = {}
        self._locked_ids: dict[int, str] = {}
        self._cum_weight: dict[int, dict[str, float]] = {}
        self._cum_count: dict[int, dict[str, int]] = {}

    def update(
        self, track_id: int | None, text: str, conf: float, valid: bool, source: str = "ocr",
    ) -> FusedTrackResult:
        if track_id is None:
            return FusedTrackResult(
                stable_id="",
                stable_conf=0.0,
                vote_ratio=0.0,
                support_count=0,
                sample_count=0,
                ready=False,
            )

        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.config.window_size)
        self._buffers[track_id].append(TrackVote(text=text, conf=conf, valid=valid, source=source))

        if valid and text:
            weight = max(0.0, conf)
            if source == "vlm_fallback":
                weight *= self.VLM_WEIGHT_BOOST
            cw = self._cum_weight.setdefault(track_id, {})
            cw[text] = cw.get(text, 0.0) + weight
            cc = self._cum_count.setdefault(track_id, {})
            cc[text] = cc.get(text, 0) + 1

        if track_id not in self._locked_ids:
            self._try_lock(track_id)

        if track_id in self._locked_ids:
            sample_count = len(self._buffers[track_id])
            return FusedTrackResult(
                stable_id=self._locked_ids[track_id],
                stable_conf=1.0,
                vote_ratio=1.0,
                support_count=self._cum_count.get(track_id, {}).get(self._locked_ids[track_id], 0),
                sample_count=sample_count,
                ready=True,
            )

        sample_count = len(self._buffers[track_id])
        cw = self._cum_weight.get(track_id, {})
        cc = self._cum_count.get(track_id, {})
        if cw:
            best_text = max(cw, key=cw.get)  # type: ignore[arg-type]
            total_weight = sum(cw.values())
            if total_weight > 1e-8:
                vote_ratio = cw[best_text] / total_weight
                support_count = cc.get(best_text, 0)
                stable_conf = cw[best_text] / max(1, support_count)
                pre_ready = support_count >= 2 and vote_ratio >= self.LOCK_MIN_RATIO
                return FusedTrackResult(
                    stable_id=best_text if pre_ready else "",
                    stable_conf=stable_conf,
                    vote_ratio=vote_ratio,
                    support_count=support_count,
                    sample_count=sample_count,
                    ready=pre_ready,
                )

        return FusedTrackResult(
            stable_id="",
            stable_conf=0.0,
            vote_ratio=0.0,
            support_count=0,
            sample_count=sample_count,
            ready=False,
        )

    def _try_lock(self, track_id: int) -> None:
        cw = self._cum_weight.get(track_id)
        cc = self._cum_count.get(track_id)
        if not cw or not cc:
            return
        best_text = max(cw, key=cw.get)  # type: ignore[arg-type]
        best_count = cc.get(best_text, 0)
        if best_count < self.LOCK_MIN_VOTES:
            return
        total_weight = sum(cw.values())
        if total_weight <= 1e-8:
            return
        if cw[best_text] / total_weight >= self.LOCK_MIN_RATIO:
            self._locked_ids[track_id] = best_text

    def _fuse(self, votes: deque[TrackVote]) -> FusedTrackResult:
        score_by_text: dict[str, float] = {}
        count_by_text: dict[str, int] = {}
        total_valid_weight = 0.0
        sample_count = len(votes)

        for vote in votes:
            if not vote.valid or not vote.text:
                continue
            weight = max(0.0, float(vote.conf))
            if vote.source == "vlm_fallback":
                weight *= self.VLM_WEIGHT_BOOST
            total_valid_weight += weight
            score_by_text[vote.text] = score_by_text.get(vote.text, 0.0) + weight
            count_by_text[vote.text] = count_by_text.get(vote.text, 0) + 1

        if not score_by_text or total_valid_weight <= 1e-8:
            return FusedTrackResult(
                stable_id="",
                stable_conf=0.0,
                vote_ratio=0.0,
                support_count=0,
                sample_count=sample_count,
                ready=False,
            )

        best_text, best_score = max(score_by_text.items(), key=lambda x: x[1])
        vote_ratio = best_score / total_valid_weight
        support_count = count_by_text.get(best_text, 0)
        stable_conf = best_score / max(1, support_count)
        ready = vote_ratio >= self.config.vote_threshold
        return FusedTrackResult(
            stable_id=best_text if ready else "",
            stable_conf=stable_conf,
            vote_ratio=vote_ratio,
            support_count=support_count,
            sample_count=sample_count,
            ready=ready,
        )
