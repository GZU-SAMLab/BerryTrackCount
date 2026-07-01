"""Reusable counters for tracked objects.

Track format: [x1, y1, x2, y2, track_id, class_id]
ID count: Count unique track IDs   
Line count: Count one occurrence when the track center crosses any counting line; the default line is located at the 50% screen position   
Area count: Count one occurrence when the track center first enters a vertical region; the default area is the middle 20%, i.e., 40%-60%
"""

from collections import defaultdict
import logging
import time
from typing import Callable, DefaultDict, Dict, Iterable, Optional, Sequence, Set, Tuple, Union


LOGGER = logging.getLogger(__name__)


class ObjectCounter:
    """Aggregate reusable counting strategies by class or stage."""

    def __init__(
        self,
        line_x: Optional[Union[int, Sequence[int]]] = None,
        area_x1: Optional[int] = None,
        area_x2: Optional[int] = None,
        label_resolver: Optional[Callable[[int], str]] = None,
    ) -> None:
        self.line_x = self._normalize_lines(line_x)
        self.area_x1 = area_x1
        self.area_x2 = area_x2
        self.label_resolver = label_resolver or (lambda class_id: str(int(class_id)))

        self.id_count: DefaultDict[str, Set[int]] = defaultdict(set)
        self.line_count: DefaultDict[str, Set[int]] = defaultdict(set)
        self.area_count: DefaultDict[str, Set[int]] = defaultdict(set)

        self._last_center_x: Dict[int, int] = {}
        self._in_area: Dict[int, bool] = {}
        self._line_seen: Set[int] = set()
        self._area_seen: Set[int] = set()
        self._id_time = 0.0
        self._line_time = 0.0
        self._area_time = 0.0
        self._id_frames = 0
        self._line_frames = 0
        self._area_frames = 0

    @staticmethod
    def _normalize_lines(line_x: Optional[Union[int, Sequence[int]]]) -> Tuple[int, ...]:
        if line_x is None:
            return ()
        if isinstance(line_x, Sequence) and not isinstance(line_x, (str, bytes)):
            return tuple(sorted({int(x) for x in line_x}))
        return (int(line_x),)

    def configure_defaults_if_needed(self, frame_width: int) -> None:
        """Fill default counting geometry from frame width."""
        if not self.line_x:
            self.line_x = (int(frame_width * 0.5),)
        if self.area_x1 is None:
            self.area_x1 = int(frame_width * 0.4)
        if self.area_x2 is None:
            self.area_x2 = int(frame_width * 0.6)
        if self.area_x1 > self.area_x2:
            self.area_x1, self.area_x2 = self.area_x2, self.area_x1

    def reset(self) -> None:
        """Clear counts and temporal states."""
        self.id_count.clear()
        self.line_count.clear()
        self.area_count.clear()
        self._last_center_x.clear()
        self._in_area.clear()
        self._line_seen.clear()
        self._area_seen.clear()
        self._id_time = 0.0
        self._line_time = 0.0
        self._area_time = 0.0
        self._id_frames = 0
        self._line_frames = 0
        self._area_frames = 0

    def update(self, tracks: Iterable[Sequence[float]], frame_size: Optional[Tuple[int, int]] = None) -> None:
        """Update all counters from one frame of tracking results."""
        if frame_size is not None:
            self.configure_defaults_if_needed(frame_size[1])

        track_items = []
        for track in tracks:
            x1, _, x2, _, track_id, class_id = track[:6]
            track_id = int(track_id)
            stage = self.label_resolver(int(class_id))
            center_x = int((float(x1) + float(x2)) * 0.5)
            track_items.append((track_id, stage, center_x))

        self._id_frames += 1
        if self.line_x:
            self._line_frames += 1
        if self.area_x1 is not None and self.area_x2 is not None:
            self._area_frames += 1

        id_start = time.perf_counter()
        for track_id, stage, center_x in track_items:
            self.id_count[stage].add(track_id)
        self._id_time += time.perf_counter() - id_start

        line_start = time.perf_counter()
        for track_id, stage, center_x in track_items:
            prev_center_x = self._last_center_x.get(track_id)
            if track_id not in self._line_seen and prev_center_x is not None:
                crossed = any(
                    (prev_center_x < line <= center_x) or (center_x <= line < prev_center_x)
                    for line in self.line_x
                )
                if crossed:
                    self.line_count[stage].add(track_id)
                    self._line_seen.add(track_id)
        if self.line_x:
            self._line_time += time.perf_counter() - line_start

        area_start = time.perf_counter()
        for track_id, stage, center_x in track_items:
            in_area = (
                self.area_x1 is not None
                and self.area_x2 is not None
                and self.area_x1 <= center_x <= self.area_x2
            )
            if track_id not in self._area_seen and in_area and not self._in_area.get(track_id, False):
                self.area_count[stage].add(track_id)
                self._area_seen.add(track_id)

            self._in_area[track_id] = in_area
            self._last_center_x[track_id] = center_x
        if self.area_x1 is not None and self.area_x2 is not None:
            self._area_time += time.perf_counter() - area_start

    @staticmethod
    def _to_counts(data: Dict[str, Set[int]]) -> Dict[str, int]:
        return {label: len(track_ids) for label, track_ids in data.items()}

    def results(self, as_counts: bool = False):
        """Return raw sets or aggregated counts."""
        if as_counts:
            return (
                self._to_counts(self.id_count),
                self._to_counts(self.line_count),
                self._to_counts(self.area_count),
            )
        return self.id_count, self.line_count, self.area_count

    def fps_results(self) -> Dict[str, float]:
        """Return per-strategy throughput in frames per second."""
        return {
            "id_fps": self._id_frames / self._id_time if self._id_frames > 0 and self._id_time > 0 else 0.0,
            "line_fps": self._line_frames / self._line_time if self._line_frames > 0 and self._line_time > 0 else 0.0,
            "area_fps": self._area_frames / self._area_time if self._area_frames > 0 and self._area_time > 0 else 0.0,
        }

    def log_results(self) -> None:
        """Log the current summary with a standard logger."""
        id_count, line_count, area_count = self.results(as_counts=True)
        fps = self.fps_results()
        LOGGER.info(
            "counter summary | id=%s | line=%s | area=%s | id_fps=%.2f | line_fps=%.2f | area_fps=%.2f",
            id_count,
            line_count,
            area_count,
            fps["id_fps"],
            fps["line_fps"],
            fps["area_fps"],
        )
