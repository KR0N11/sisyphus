"""WR-pace ghost: the benchmark run every Mario races against.

Loads a frame-indexed x-position trace from data/ghost_1_1.json. The bundled
trace is a synthetic max-run-speed model of world-record pace; a real WR/TAS
trace in the same JSON format is a drop-in replacement.
"""

import json
from pathlib import Path

GHOST_PATH = Path(__file__).resolve().parents[1] / "data" / "ghost_1_1.json"


class Ghost:
    def __init__(self, path: Path = GHOST_PATH):
        data = json.loads(Path(path).read_text())
        self.frames: list[float] = data["frames"]
        self.start_x: float = data["start_x"]
        self.flag_x: float = data["flag_x"]
        self.finish_frame: int = data["finish_frame"]
        self.finish_time_s: float = data["finish_time_s"]
        self.source: str = data["source"]

    def x_at(self, frame: int) -> float:
        """Ghost x-position at a given episode frame (parked at flag after finish)."""
        if frame >= len(self.frames):
            return self.flag_x
        return self.frames[frame]

    def progress_at(self, frame: int) -> float:
        """0..1 fraction of the level the ghost has covered at this frame."""
        return (self.x_at(frame) - self.start_x) / (self.flag_x - self.start_x)
