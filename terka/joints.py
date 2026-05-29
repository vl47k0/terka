"""MediaPipe Pose Landmarker → rakija PoseJoints mapping.

Same mapping kadar's RigJoints.kt does, in Python. Names match
rakija's PoseJoints struct field names exactly — that's the data
contract on disk; rakija's pose_rig_trajectory_load_json reads
them back unmodified.

MediaPipe world landmarks are in metres, hip-centered, with axes
that follow the camera. We dump them as-is (no calibration
transform) — rakija renders the figure in some camera-relative
orientation, but the per-joint trail shape + the swing motion
are intact, which is what we want for the THETIS comparison
flow. Future T-pose calibration (like kadar's) could slot in here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

# MediaPipe 33-landmark COCO-ish indices used by Pose Landmarker.
NOSE             =  0
LEFT_SHOULDER    = 11
RIGHT_SHOULDER   = 12
LEFT_ELBOW       = 13
RIGHT_ELBOW      = 14
LEFT_WRIST       = 15
RIGHT_WRIST      = 16
LEFT_HIP         = 23
RIGHT_HIP        = 24
LEFT_KNEE        = 25
RIGHT_KNEE       = 26
LEFT_ANKLE       = 27
RIGHT_ANKLE      = 28
LEFT_FOOT_INDEX  = 31
RIGHT_FOOT_INDEX = 32


@dataclass
class RigJoints:
    """One frame of joint positions in the rakija PoseJoints schema.

    Field names match the C struct exactly (snake_case) — the
    trajectory serialiser writes them straight to JSON.
    """
    pelvis:       List[float]
    spine_top:    List[float]
    head_center:  List[float]
    l_hip:        List[float]
    r_hip:        List[float]
    l_knee:       List[float]
    r_knee:       List[float]
    l_ankle:      List[float]
    r_ankle:      List[float]
    l_toe:        List[float]
    r_toe:        List[float]
    l_shoulder:   List[float]
    r_shoulder:   List[float]
    r_elbow:      List[float]
    r_wrist:      List[float]
    r_racket_tip: List[float]
    l_elbow:      List[float]
    l_hand:       List[float]


def _vec(lm) -> List[float]:
    return [float(lm.x), float(lm.y), float(lm.z)]


def _mid(a, b) -> List[float]:
    return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]


def mediapipe_to_joints(world_landmarks) -> RigJoints | None:
    """Map a MediaPipe world-landmark list (33 entries) to RigJoints.

    Returns None if the landmark list is shorter than expected
    (MediaPipe didn't track all the keypoints — rare but happens
    when limbs leave the frame).
    """
    if len(world_landmarks) < 33:
        return None

    nose  = _vec(world_landmarks[NOSE])
    l_hip = _vec(world_landmarks[LEFT_HIP])
    r_hip = _vec(world_landmarks[RIGHT_HIP])
    l_sh  = _vec(world_landmarks[LEFT_SHOULDER])
    r_sh  = _vec(world_landmarks[RIGHT_SHOULDER])
    r_wr  = _vec(world_landmarks[RIGHT_WRIST])

    return RigJoints(
        pelvis      = _mid(l_hip, r_hip),
        spine_top   = _mid(l_sh, r_sh),
        head_center = nose,
        l_hip       = l_hip,
        r_hip       = r_hip,
        l_knee      = _vec(world_landmarks[LEFT_KNEE]),
        r_knee      = _vec(world_landmarks[RIGHT_KNEE]),
        l_ankle     = _vec(world_landmarks[LEFT_ANKLE]),
        r_ankle     = _vec(world_landmarks[RIGHT_ANKLE]),
        l_toe       = _vec(world_landmarks[LEFT_FOOT_INDEX]),
        r_toe       = _vec(world_landmarks[RIGHT_FOOT_INDEX]),
        l_shoulder  = l_sh,
        r_shoulder  = r_sh,
        r_elbow     = _vec(world_landmarks[RIGHT_ELBOW]),
        r_wrist     = r_wr,
        # No racket marker on THETIS videos — tip stubs to the
        # wrist, same as kadar's pre-marker MVP. The shape of the
        # arm chain still tells you what the swing is doing.
        r_racket_tip = r_wr,
        l_elbow      = _vec(world_landmarks[LEFT_ELBOW]),
        l_hand       = _vec(world_landmarks[LEFT_WRIST]),
    )
