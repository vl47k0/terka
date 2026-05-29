"""MediaPipe Pose Landmarker driver — frame iterator over a video file.

OpenCV reads each AVI frame, MediaPipe consumes them in VIDEO
running-mode (synchronous detectForVideo) and yields one RigJoints
per detection. Frames where MediaPipe loses the player drop out
silently; the caller decides whether to interpolate or just write
a shorter trajectory.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2  # type: ignore[import-not-found]
import mediapipe as mp  # type: ignore[import-not-found]
from mediapipe.tasks import python as mp_python  # type: ignore[import-not-found]
from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import-not-found]

from terka.joints import RigJoints, mediapipe_to_joints


@contextmanager
def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"could not open {path}")
    try:
        yield cap
    finally:
        cap.release()


def video_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    return float(fps) if fps > 0 else 30.0


@contextmanager
def make_landmarker(model_path: Path):
    """Build a Pose Landmarker in VIDEO mode + tear it down."""
    base = mp_python.BaseOptions(model_asset_path=str(model_path))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    lm = mp_vision.PoseLandmarker.create_from_options(opts)
    try:
        yield lm
    finally:
        lm.close()


def detect_with_landmarker(
    video_path: Path,
    landmarker,
    *,
    session_offset_ms: int = 0,
) -> Iterator[tuple[float, RigJoints]]:
    """Yield (t_local_seconds, RigJoints) for every detected frame.

    The landmarker is supplied by the caller — batch ingest creates
    one and reuses it across the whole run, saving the ~300 ms
    EGL + model-load overhead the per-video path would otherwise
    repeat 1980 times.

    MediaPipe's VIDEO running-mode contract requires monotonically
    increasing timestamps. Each video's own CAP_PROP_POS_MSEC
    resets to 0, so we add `session_offset_ms` per frame to keep
    the sequence monotone across videos. The yielded `t` stays
    per-video so the resulting trajectories still start at 0
    (rakija expects local-to-the-swing times).
    """
    with open_video(video_path) as cap:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            t_local_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # OpenCV → MediaPipe wants RGB.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame_rgb,
            )
            result = landmarker.detect_for_video(
                mp_image, int(session_offset_ms + t_local_ms),
            )
            if not result.pose_world_landmarks:
                continue
            joints = mediapipe_to_joints(result.pose_world_landmarks[0])
            if joints is None:
                continue
            yield t_local_ms / 1000.0, joints


def detect_video(video_path: Path, model_path: Path,
                 ) -> Iterator[tuple[float, RigJoints]]:
    """Single-video convenience: build a fresh landmarker, run, tear
    down. Used by the `convert` subcommand. The `ingest` batch path
    uses `detect_with_landmarker` directly with a shared instance.
    """
    with make_landmarker(model_path) as lm:
        yield from detect_with_landmarker(video_path, lm)
