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


def detect_video(video_path: Path, model_path: Path,
                 ) -> Iterator[tuple[float, RigJoints]]:
    """Yield (timestamp_seconds, RigJoints) for every frame where
    MediaPipe successfully landmarked the player.

    Timestamps come from CAP_PROP_POS_MSEC so they reflect the
    actual frame timing in the source video (THETIS recordings are
    17 fps but we don't hardcode that — different sources will
    have different rates).
    """
    with open_video(video_path) as cap, make_landmarker(model_path) as lm:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # OpenCV → MediaPipe wants RGB.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame_rgb,
            )
            result = lm.detect_for_video(mp_image, int(t_ms))
            if not result.pose_world_landmarks:
                continue
            joints = mediapipe_to_joints(result.pose_world_landmarks[0])
            if joints is None:
                continue
            yield t_ms / 1000.0, joints
