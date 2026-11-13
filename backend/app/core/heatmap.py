"""HeatmapRenderer - overlays LIME importance / error maps / highlighted regions on the frame
(Section 4.4, Figures 6.6 - 6.9). All maps are computed at model resolution (128 x 128) and
up-sampled to the display frame, so they stay aligned with the (squashed) network input."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import mark_boundaries

DISPLAY_WIDTH = 480


def to_display(frame_rgb: np.ndarray, width: int = DISPLAY_WIDTH) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if w == width:
        return frame_rgb
    return cv2.resize(frame_rgb, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def _up(arr: np.ndarray, size_hw: tuple[int, int], nearest: bool = False) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def _jet(norm01: np.ndarray) -> np.ndarray:
    u8 = np.clip(norm01 * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def error_map(frame128: np.ndarray, recon128: np.ndarray, display_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Squared difference between frame and reconstruction -> (jet image, raw float map at 128)."""
    err = ((frame128.astype(np.float32) / 255.0 - recon128.astype(np.float32) / 255.0) ** 2).mean(axis=2)
    norm = (err - err.min()) / (err.max() - err.min() + 1e-8)
    return _jet(_up(norm, display_hw)), err


def lime_heatmap(display: np.ndarray, segments: np.ndarray, weights: dict[int, float], alpha: float = 0.55) -> np.ndarray:
    """Red = high positive importance, blue = low / negative (Figure 6.8)."""
    wmap = np.zeros(segments.shape, dtype=np.float32)
    for s, w in weights.items():
        wmap[segments == s] = w
    lo, hi = float(wmap.min()), float(wmap.max())
    norm = (wmap - lo) / (hi - lo + 1e-12)
    heat = _jet(_up(norm, display.shape[:2]))
    return cv2.addWeighted(display, 1 - alpha, heat, alpha, 0)


def superpixel_image(display: np.ndarray, segments: np.ndarray) -> np.ndarray:
    seg = _up(segments.astype(np.int32).astype(np.float32), display.shape[:2], nearest=True).astype(np.int32)
    rng = np.random.RandomState(1)
    palette = rng.randint(60, 230, size=(int(seg.max()) + 1, 3)).astype(np.uint8)
    colour = palette[seg]
    blended = cv2.addWeighted(display, 0.55, colour, 0.45, 0)
    out = mark_boundaries(blended, seg, color=(1, 1, 1), mode="inner")
    return (out * 255).astype(np.uint8)


def bbox_from_mask(mask128: np.ndarray, pad: int = 2) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask128)
    if xs.size == 0:
        return None
    H, W = mask128.shape
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(W, int(xs.max()) + 1 + pad), min(H, int(ys.max()) + 1 + pad))


def bbox_from_error(err128: np.ndarray) -> tuple[int, int, int, int]:
    """Fallback box: bounding box of the top-error blob when LIME is not available."""
    blur = cv2.GaussianBlur(err128.astype(np.float32), (0, 0), 3)
    mask = blur >= np.percentile(blur, 97)
    return bbox_from_mask(mask) or (0, 0, err128.shape[1], err128.shape[0])


def draw_box(display: np.ndarray, box128: tuple[int, int, int, int], colour=(255, 40, 40), thickness: int = 2) -> np.ndarray:
    H, W = display.shape[:2]
    x0, y0, x1, y1 = box128
    sx, sy = W / 128.0, H / 128.0
    p0, p1 = (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy))
    out = display.copy()
    cv2.rectangle(out, p0, p1, colour, thickness)
    return out


def zoom_crop(display: np.ndarray, box128: tuple[int, int, int, int], out_w: int = 360) -> np.ndarray:
    H, W = display.shape[:2]
    x0, y0, x1, y1 = box128
    sx, sy = W / 128.0, H / 128.0
    crop = display[int(y0 * sy):max(int(y1 * sy), int(y0 * sy) + 2), int(x0 * sx):max(int(x1 * sx), int(x0 * sx) + 2)]
    h, w = crop.shape[:2]
    scale = out_w / max(w, 1)
    return cv2.resize(crop, (out_w, max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)


def save_jpg(path, rgb: np.ndarray, quality: int = 88) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
