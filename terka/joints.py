"""MediaPipe Pose Landmarker → rakija PoseJoints mapping.

Same mapping kadar's RigJoints.kt does, in Python. Names match
rakija's PoseJoints struct field names exactly — that's the data
contract on disk; rakija's pose_rig_trajectory_load_json reads
them back unmodified.

Axis convention
---------------
MediaPipe world landmarks are in metres, hip-centered, with
camera-frame axes:
  +x = image right          (subject's anatomical LEFT  when facing camera)
  +y = image down
  +z = away from the camera
  origin = midpoint of the hips

rakija's body canvas expects:
  +x = subject's anatomical-forward      (facing direction, "downrange")
  +y = up                                (body_project: sy = h/2 − y·scale)
  +z = subject's anatomical-left         (so −z = anatomical-right; this
                                          is what body_rig.c places
                                          r_hip at when shoulder_yaw=0,
                                          confirmed by the default
                                          forehand prep)
  origin = floor; pelvis anchored at y ≈ 1.10

We bridge them with a SINGLE composed transform applied per joint:

  rakija_x =  −mp_z          # MP "into scene"  → rakija forward
  rakija_y =  −mp_y + 1.10   # MP "down"        → rakija up + lift pelvis
  rakija_z =  +mp_x          # MP "image right" → rakija anatomical-left
                             #   (= subject's left when facing camera)

Decomposes as: (1) negate y to flip MP's image-down to rakija's
up; (2) lift y by 1.10 to put the pelvis on rakija's canvas
anchor; (3) rotate 90° about +y so the player's facing direction
(toward the camera in THETIS = MP −z) aligns with rakija +x. All
three operations together are proper (preserve handedness), so
pose_from_joints back-solves cleanly when the user toggles
"Drive Skeleton from Trail".

Sanity check on a THETIS forehand contact frame
(MP r_wrist ≈ (−0.11, −0.31, −0.52) — wrist reaches toward the
camera as the player swings):
  rakija_x = +0.52  (out front, "downrange")     ✓
  rakija_y =  1.41  (chest-height above pelvis)  ✓
  rakija_z = −0.11  (slightly to subject's right) ✓

Racket-tip extrapolation
------------------------
THETIS has no racket marker. r_racket_tip stubbed to r_wrist is
visually misleading because the racket head extends ~55 cm past
the wrist — a forehand-contact "tip" really sits 50 cm higher /
further than the wrist alone. We extrapolate the tip along the
forearm direction (elbow → wrist, normalised, scaled by
RACKET_HEAD_OFFSET_M). Accurate when the wrist is locked
(forearm + racket collinear); under-rotated when the wrist
breaks (slice, kick serve) — but the swing SHAPE comes through.

Future kadar capture with an ArUco marker on the racket throat
replaces this guess with measured tip pose; the data contract
stays identical.

What this does NOT fix
- Subject orientation in the THETIS recording is camera-relative,
  not court-relative. If a clip isn't square-to-camera (some
  serves stand sideways), our +90° rotation puts them facing
  somewhere odd in rakija world. Per-clip T-pose calibration
  (like kadar's) is the proper fix; a sweep over the corpus
  could detect the dominant facing direction and rotate
  per-clip. Out of scope for the THETIS bulk import.
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


# Rakija anchors its body view on world y≈1.10 (the canonical
# pelvis height in body_project). MediaPipe puts pelvis at y=0
# (hip-centered), so every joint gets lifted by this constant to
# make the figure stand at the right height on the canvas.
_PELVIS_LIFT_Y = 1.10

# Distance from the wrist to the racket head tip along the
# forearm direction. Adult racket length is ~68.5 cm head-to-butt;
# the wrist holds the grip ~10-15 cm from the butt, leaving ~55 cm
# of head extending past the wrist. This is the "best guess" we
# use because THETIS has no racket marker — see module docstring.
_RACKET_HEAD_OFFSET_M = 0.55


def _vec(lm) -> List[float]:
    # MP camera frame → rakija body frame. See module docstring
    # for the derivation; this single composed line is the only
    # bridge between the two coordinate systems and the only
    # place to change if a different orientation is needed.
    return [
        -float(lm.z),
        -float(lm.y) + _PELVIS_LIFT_Y,
        +float(lm.x),
    ]


def _racket_tip_from(wrist: List[float], elbow: List[float]) -> List[float]:
    """Extrapolate racket-tip = wrist + offset · normalize(wrist − elbow).

    The forearm direction is the closest thing we have to a racket
    direction without a marker on the racket. Degenerate input
    (wrist == elbow) falls back to the wrist position so the trail
    doesn't NaN out.
    """
    fx = wrist[0] - elbow[0]
    fy = wrist[1] - elbow[1]
    fz = wrist[2] - elbow[2]
    mag = (fx * fx + fy * fy + fz * fz) ** 0.5
    if mag < 1e-6:
        return [wrist[0], wrist[1], wrist[2]]
    s = _RACKET_HEAD_OFFSET_M / mag
    return [wrist[0] + fx * s, wrist[1] + fy * s, wrist[2] + fz * s]


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
    r_el  = _vec(world_landmarks[RIGHT_ELBOW])
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
        r_elbow     = r_el,
        r_wrist     = r_wr,
        # No racket marker on THETIS — extrapolate the tip along
        # the forearm direction so the trail reads as a real
        # racket head sweeping through the swing instead of a
        # short wrist trail. See module docstring.
        r_racket_tip = _racket_tip_from(r_wr, r_el),
        l_elbow      = _vec(world_landmarks[LEFT_ELBOW]),
        l_hand       = _vec(world_landmarks[LEFT_WRIST]),
    )
