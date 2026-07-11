"""Video-level TIM metrics (analogue of token logprob drift for Wan2.1)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

def _load_frames_opencv(path: Path) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"opencv cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames in {path}")
    return np.stack(frames, axis=0).astype(np.float32) / 255.0


def _load_frames_imageio(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    reader = imageio.get_reader(path, format="ffmpeg")
    frames = [np.asarray(frame, dtype=np.float32) / 255.0 for frame in reader]
    reader.close()
    if not frames:
        raise ValueError(f"No frames in {path}")
    return np.stack(frames, axis=0)


def _load_frames(path: Path, max_frames: int | None = None) -> np.ndarray:
    errors: list[str] = []

    for loader in (_load_frames_opencv, _load_frames_imageio):
        try:
            frames = loader(path)
            break
        except Exception as exc:  # noqa: BLE001 — try next backend
            errors.append(f"{loader.__name__}: {exc}")
    else:
        raise ImportError(
            "Could not read video. Install opencv-python or imageio-ffmpeg.\n"
            + "\n".join(errors)
        )

    if frames.ndim == 3:
        frames = frames[None, ...]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


@dataclass
class TimVideoReport:
    """Compare two videos generated from the same prompt + seed."""

    video_a: str
    video_b: str
    profile_a: str
    profile_b: str
    num_frames_compared: int
    mean_frame_mse: float
    max_frame_mse: float
    late_frame_mse_mean: float  # last 30% frames — chaotic amplification proxy
    per_frame_mse: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def severity(self) -> str:
        if self.max_frame_mse > 0.05 or self.late_frame_mse_mean > 2 * self.mean_frame_mse:
            return "warning"
        if self.mean_frame_mse < 1e-4:
            return "ok"
        return "info"


def compare_videos(
    path_a: Path | str,
    path_b: Path | str,
    *,
    profile_a: str = "a",
    profile_b: str = "b",
) -> TimVideoReport:
    """Pixel MSE per frame between two MP4s (same length required)."""
    path_a, path_b = Path(path_a), Path(path_b)
    fa = _load_frames(path_a)
    fb = _load_frames(path_b)
    n = min(len(fa), len(fb))
    if n == 0:
        raise ValueError("No frames to compare")
    fa, fb = fa[:n], fb[:n]

    mse_per_frame = [float(np.mean((fa[i] - fb[i]) ** 2)) for i in range(n)]
    late_start = max(int(n * 0.7), 0)

    return TimVideoReport(
        video_a=str(path_a),
        video_b=str(path_b),
        profile_a=profile_a,
        profile_b=profile_b,
        num_frames_compared=n,
        mean_frame_mse=float(np.mean(mse_per_frame)),
        max_frame_mse=float(np.max(mse_per_frame)),
        late_frame_mse_mean=float(np.mean(mse_per_frame[late_start:])),
        per_frame_mse=mse_per_frame,
    )


def run_generate(
    repo_root: Path,
    profile_argv: list[str],
    *,
    prompt: str,
    seed: int,
    output: Path,
) -> None:
    """Invoke ./generate.sh with a profile."""
    cmd = [
        str(repo_root / "generate.sh"),
        *profile_argv,
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)


def save_report(report: TimVideoReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))
