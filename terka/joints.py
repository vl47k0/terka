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

Smoothing + swing-window trim
-----------------------------
MediaPipe LITE on 640×480 Kinect video is noisy — even when the
subject is standing still, joints wobble ~3 cm/frame (median
trail-segment jump, measured on p32 forehand). Across a 4.7 s
THETIS clip with only ~0.5 s of actual swing, that noise
dominates the visualisation and the trail reads as a scribble
instead of a swing.

`smooth_samples` runs a 3-frame box average over all joint
positions (boundary frames use one-sided windows). 3 frames
≈ 175 ms — small enough to preserve the crisp velocity peak
at contact, large enough to wipe out the static-frame jitter.

`trim_to_swing_window` finds the peak wrist velocity frame
(after smoothing — noise spikes shouldn't pick the peak) and
keeps `[i_peak − 6, i_peak + 10]` ≈ 0.35 s prep + 0.6 s
follow-through. Sets `duration_s` from the trimmed window so
playback runs in real time.

Both ops are post-processing on the assembled per-frame
RigJoints list, not inside `mediapipe_to_joints`, so the
mapping stays single-responsibility (MP → rakija axes) and
the cleanup is testable in isolation.

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
    #
    # The z sign is empirically chosen: with +mp.x, forehand and
    # backhand swept in the same screen-x direction (visible in
    # rakija). Flipping to -mp.x mirrors them, so a right-handed
    # forehand sweeps opposite the right-handed backhand as
    # anatomically expected.
    return [
        -float(lm.z),
        -float(lm.y) + _PELVIS_LIFT_Y,
        -float(lm.x),
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
        # Placeholder — the racket-tip extrapolation happens AFTER
        # smoothing so the tip is computed from the smoothed
        # forearm direction (otherwise the smoothed wrist+elbow
        # would disagree with the un-smoothed tip).
        r_racket_tip = [r_wr[0], r_wr[1], r_wr[2]],
        l_elbow      = _vec(world_landmarks[LEFT_ELBOW]),
        l_hand       = _vec(world_landmarks[LEFT_WRIST]),
    )


# --------------------------------------------------------------------- #
# Post-processing: jitter smoothing + swing-window trim                  #
# --------------------------------------------------------------------- #
#
# Operates on the assembled per-frame (t, RigJoints) list produced by
# detect_video / detect_with_landmarker. Keeping these out of
# mediapipe_to_joints means the MP→rakija mapping stays single-
# responsibility, and the cleanup is unit-testable in isolation.


_JOINT_FIELDS = (
    "pelvis", "spine_top", "head_center",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
    "l_toe", "r_toe", "l_shoulder", "r_shoulder",
    "r_elbow", "r_wrist", "r_racket_tip",
    "l_elbow", "l_hand",
)


def smooth_samples(samples, window: int = 3):
    """3-frame box average on every joint position (boundaries one-sided).

    Wipes the ~3 cm/frame jitter MediaPipe LITE adds to static joints
    without blunting the swing's velocity peak (3 frames ≈ 175 ms at
    17 fps, much shorter than a swing's ~500 ms duration). `window`
    must be odd; default 3 is what bulk-import needs. Returns a new
    list — input is not mutated.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and ≥ 1, got {window}")
    if len(samples) < 2:
        return list(samples)

    half = window // 2
    n = len(samples)
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        k = hi - lo
        new_joints = {}
        for field in _JOINT_FIELDS:
            sx = sy = sz = 0.0
            for j in range(lo, hi):
                v = getattr(samples[j][1], field)
                sx += v[0]; sy += v[1]; sz += v[2]
            new_joints[field] = [sx / k, sy / k, sz / k]
        out.append((samples[i][0], RigJoints(**new_joints)))
    return out


def apply_racket_tip(samples):
    """Recompute r_racket_tip from each frame's (smoothed) elbow + wrist.

    Run AFTER smooth_samples so the tip uses the smoothed forearm
    direction. Mutates the existing RigJoints in place — cheap, and
    sample lists are owned by the per-video pipeline.
    """
    for _, j in samples:
        j.r_racket_tip = _racket_tip_from(j.r_wrist, j.r_elbow)
    return samples


def trim_to_swing_window(samples, prep_frames: int = 6,
                         follow_frames: int = 10):
    """Trim the trajectory to a window around the peak wrist velocity.

    THETIS clips are ~4.7 s but only ~0.5 s is the actual swing —
    the rest is the subject standing still with MediaPipe wobbling
    their joints. Keeping the full clip floods the trail with
    static-frame noise. Default window: 6 frames prep + 10 frames
    follow-through ≈ 0.35 s + 0.6 s at 17 fps.

    The trimmed sample timestamps are RE-BASED to start at t=0 so
    rakija's scrub slider covers the swing window, not the original
    setup-padded range. Returns (new_samples, new_duration_s).
    """
    if len(samples) < 3:
        # Too short to find a peak meaningfully — return as-is.
        if not samples:
            return samples, 0.0
        return list(samples), samples[-1][0] - samples[0][0]

    n = len(samples)
    peak_i = 0
    peak_v2 = -1.0
    # Central-difference wrist velocity; skip endpoints.
    for i in range(1, n - 1):
        t_prev, j_prev = samples[i - 1]
        t_next, j_next = samples[i + 1]
        dt = t_next - t_prev
        if dt <= 0:
            continue
        dx = (j_next.r_wrist[0] - j_prev.r_wrist[0]) / dt
        dy = (j_next.r_wrist[1] - j_prev.r_wrist[1]) / dt
        dz = (j_next.r_wrist[2] - j_prev.r_wrist[2]) / dt
        v2 = dx * dx + dy * dy + dz * dz
        if v2 > peak_v2:
            peak_v2 = v2
            peak_i = i

    lo = max(0, peak_i - prep_frames)
    hi = min(n, peak_i + follow_frames + 1)
    window = list(samples[lo:hi])
    t0 = window[0][0]
    rebased = [(t - t0, j) for (t, j) in window]
    duration = rebased[-1][0] - rebased[0][0]
    return rebased, duration
