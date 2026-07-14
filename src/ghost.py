"""WR-pace ghost: the benchmark run every Mario races against.

Loads a frame-indexed x-position trace from data/ghost_1_1.json. The bundled
trace is a synthetic max-run-speed model of world-record pace; a real WR/TAS
trace in the same JSON format is a drop-in replacement.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GHOST_PATH = DATA_DIR / "ghost_1_1.json"


def ghost_path(level: str) -> Path:
    return DATA_DIR / f"ghost_{level.replace('-', '_')}.json"


def for_level(level: str):
    """Ghost for this level, or None when no trace has been recorded."""
    path = ghost_path(level)
    return Ghost(path) if path.exists() else None


GROUND_TOP = 192  # screen y of a grounded small Mario's sprite top
GAME_FPS = 50     # PAL-physics ROM: real-world seconds = frames / 50


class Ghost:
    def __init__(self, path: Path = GHOST_PATH):
        data = json.loads(Path(path).read_text())
        self.frames: list[float] = data["frames"]
        self.y_top: list[int] | None = data.get("y_top")
        self.start_x: float = data["start_x"]
        self.flag_x: float = data["flag_x"]
        self.finish_frame: int = data["finish_frame"]
        # recomputed at load so old traces stay valid after the fps correction
        self.finish_time_s: float = round(self.finish_frame / GAME_FPS, 2)
        self.source: str = data["source"]

    def x_at(self, frame: int) -> float:
        """Ghost x-position at a given episode frame (parked at flag after finish)."""
        if frame >= len(self.frames):
            return self.flag_x
        return self.frames[frame]

    def y_top_at(self, frame: int) -> int:
        """Screen y of the ghost sprite's top edge (jumps included when the
        trace has y data; grounded otherwise)."""
        if self.y_top is None or frame >= len(self.y_top):
            return GROUND_TOP
        return self.y_top[frame]

    def progress_at(self, frame: int) -> float:
        """0..1 fraction of the level the ghost has covered at this frame."""
        return (self.x_at(frame) - self.start_x) / (self.flag_x - self.start_x)
