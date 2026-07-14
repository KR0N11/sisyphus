"""HUD composition: tiles every emulator frame into one mosaic with a status
header and a race bar showing the current leader against the WR ghost.

Tile borders are green when that Mario is ahead of ghost pace at the same
episode frame, red when behind.
"""

import math
from pathlib import Path

import cv2
import numpy as np

TILE_W, TILE_H = 256, 240
HEADER_H, RACE_H = 44, 60
FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN = (60, 200, 60)
RED = (50, 50, 220)
WHITE = (240, 240, 240)
GRAY = (90, 90, 90)

GHOST_OPACITY = 0.45
_SPRITE_PATH = Path(__file__).resolve().parents[1] / "data" / "ghost_sprite.png"
_sprite_cache: list = []


def _ghost_sprite():
    """RGBA ghost sprite (BGR whitened for a spectral look), loaded once."""
    if not _sprite_cache:
        rgba = cv2.imread(str(_SPRITE_PATH), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            _sprite_cache.append(None)
        else:
            bgr = (rgba[:, :, :3].astype(np.float32) * 0.55 + 255 * 0.45)
            alpha = rgba[:, :, 3].astype(np.float32) / 255.0 * GHOST_OPACITY
            _sprite_cache.append((bgr, alpha[:, :, None]))
    return _sprite_cache[0]


def _overlay_ghost(tile, ghost_x, ghost_top_y, cam_x):
    """Blend the translucent pro-Mario at the ghost's world x/y on this tile."""
    sprite = _ghost_sprite()
    if sprite is None:
        return
    bgr, alpha = sprite
    h, w = bgr.shape[:2]
    sx, sy = int(ghost_x - cam_x), int(ghost_top_y)
    if sx + w <= 0 or sx >= TILE_W or sy + h <= 0 or sy >= TILE_H:
        return
    x0, x1 = max(sx, 0), min(sx + w, TILE_W)
    y0, y1 = max(sy, 0), min(sy + h, TILE_H)
    cx0, cx1, cy0, cy1 = x0 - sx, x1 - sx, y0 - sy, y1 - sy
    region = tile[y0:y1, x0:x1].astype(np.float32)
    a = alpha[cy0:cy1, cx0:cx1]
    tile[y0:y1, x0:x1] = (region * (1 - a)
                          + bgr[cy0:cy1, cx0:cx1] * a).astype(np.uint8)


def compose(frames, states, ghost, stats):
    """frames: list of RGB arrays; states: per-env dicts {x, frame};
    stats: {steps, episodes, flags, best} for the header."""
    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.zeros((rows * TILE_H, cols * TILE_W, 3), np.uint8)

    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        tile = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
        if tile.shape[:2] != (TILE_H, TILE_W):
            tile = cv2.resize(tile, (TILE_W, TILE_H))
        s = states[i]
        ghost_x = ghost.x_at(s["frame"])
        delta = s["x"] - ghost_x
        _overlay_ghost(tile, ghost_x, ghost.y_top_at(s["frame"]),
                       s["x"] - s.get("screen_x", 0))
        color = GREEN if delta >= 0 else RED
        cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), color, 3)
        sign = "+" if delta >= 0 else ""
        cv2.rectangle(tile, (3, TILE_H - 26), (TILE_W - 4, TILE_H - 4), (15, 15, 15), -1)
        cv2.putText(tile, f"#{i}  x={s['x']}  {sign}{int(delta)} vs ghost",
                    (8, TILE_H - 10), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
        grid[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile

    header = np.full((HEADER_H, grid.shape[1], 3), 28, np.uint8)
    best = stats.get("best")
    best_txt = f"{best:.2f}s" if best is not None else "none yet"
    scale = 0.62 if grid.shape[1] >= 1000 else 0.42
    cv2.putText(header,
                f"MARIO RL vs WR GHOST ({ghost.finish_time_s:.2f}s)   "
                f"steps {stats.get('steps', 0):,}   episodes {stats.get('episodes', 0)}   "
                f"flags {stats.get('flags', 0)}   best clear: {best_txt}",
                (12, 29), FONT, scale, WHITE, 1, cv2.LINE_AA)

    return np.vstack([header, grid, _race_bar(grid.shape[1], states, ghost)])


def with_banner(img, text):
    """Dim a center strip and print text over it (shown during PPO updates)."""
    out = img.copy()
    h, w = out.shape[:2]
    y0, y1 = h // 2 - 36, h // 2 + 36
    out[y0:y1] = (out[y0:y1].astype(np.float32) * 0.25).astype(np.uint8)
    size = cv2.getTextSize(text, FONT, 1.0, 2)[0]
    cv2.putText(out, text, ((w - size[0]) // 2, h // 2 + 12),
                FONT, 1.0, WHITE, 2, cv2.LINE_AA)
    return out


def _race_bar(width, states, ghost):
    """One lane: the current leader's marker vs where the ghost is at the
    leader's episode frame. The finish line is the flagpole."""
    img = np.full((RACE_H, width, 3), 18, np.uint8)
    left, right = 70, width - 70
    y = RACE_H // 2 + 6

    def px(x):
        frac = (x - ghost.start_x) / (ghost.flag_x - ghost.start_x)
        return int(left + min(max(frac, 0.0), 1.0) * (right - left))

    cv2.line(img, (left, y), (right, y), GRAY, 2)
    cv2.line(img, (right, y - 12), (right, y + 12), WHITE, 2)  # finish line
    cv2.putText(img, "FLAG", (right - 18, y - 16), FONT, 0.4, WHITE, 1, cv2.LINE_AA)

    lead_i = max(range(len(states)), key=lambda i: states[i]["x"])
    lead = states[lead_i]
    gx = ghost.x_at(lead["frame"])

    cv2.circle(img, (px(gx), y), 7, WHITE, -1)
    cv2.putText(img, "GHOST", (min(px(gx) - 22, right - 95), y - 14),
                FONT, 0.4, WHITE, 1, cv2.LINE_AA)
    cv2.circle(img, (px(lead["x"]), y), 7, GREEN, -1)
    cv2.putText(img, f"LEADER #{lead_i}", (max(px(lead["x"]) - 34, 2), y + 26),
                FONT, 0.4, GREEN, 1, cv2.LINE_AA)
    return img
