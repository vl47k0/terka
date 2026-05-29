# terka

THETIS → rakija pose-trajectory converter. Walks the
[THETIS dataset](http://thetis.image.ece.ntua.gr/) (1980 Kinect
recordings of 12 tennis-shot classes across 55 subjects, 31
beginners + 24 experts), runs MediaPipe Pose Landmarker on each
RGB video, maps the 33 landmarks to rakija's 18-joint PoseJoints
schema, and POSTs the resulting trajectory JSON to vertex's
`/trajectories/` endpoint. rakija picks each one up via its
existing "Load Latest from Vertex" right-click flow.

THETIS itself doesn't ship per-frame joint coordinates — the
`VIDEO_Skelet3D/` directory contains rendered stick-figure AVIs,
not data. terka extracts coords by running pose estimation on the
RGB videos; same MediaPipe pipeline kadar uses, just on
pre-recorded video instead of a live phone camera.

## Setup

```sh
cd projects/terka
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Grab a Pose Landmarker model file (LITE ~3 MB; HEAVY ~26 MB
# but ~5× slower per frame — for batch import LITE is usually
# fine). Lives outside the repo (gitignored).
.venv/bin/python -m terka download-model --variant lite \
    --dest ./pose_landmarker.task
```

## Single-video smoke test

```sh
.venv/bin/python -m terka convert \
    ../../ref/dataset/VIDEO_RGB/forehand_flat/p32_foreflat_s1.avi \
    --model ./pose_landmarker.task \
    --pretty -o /tmp/p32_foreflat_s1.json
```

Then drop the JSON in rakija:

```sh
cp /tmp/p32_foreflat_s1.json ~/.config/rakija/trajectory.json
# In rakija: right-click on Body → Load Trail from JSON → Drive
# Skeleton from Trail.
```

## Batch import to vertex

```sh
# Start vertex first.
cd ../vertex && .venv/bin/python manage.py runserver 0.0.0.0:8000 &
cd ../terka

# Smoke-test with a 3-video sample.
.venv/bin/python -m terka ingest \
    ../../ref/dataset/VIDEO_RGB \
    --model ./pose_landmarker.task \
    --vertex-url http://127.0.0.1:8000 \
    --actions forehand_flat \
    --limit 3

# Or import everything.
.venv/bin/python -m terka ingest \
    ../../ref/dataset/VIDEO_RGB \
    --model ./pose_landmarker.task \
    --vertex-url http://127.0.0.1:8000
```

Expected runtime on the full 1980-video dataset:
- LITE model: ~25–30 minutes
- HEAVY model: ~2–3 hours

Each video appears in vertex's `/admin/` with a `device_id` like
`thetis:expert:p32:forehand_flat:s1` — substring-filterable in
the admin list view by expertise / subject / action / sequence.

The trajectory payload also carries a `thetis` block at top level
holding the parsed metadata; rakija's loader ignores it (only
reads `samples` + `duration_s`) but it stays available in vertex
for any downstream analysis that wants subject + action without
re-parsing the device_id string.

## Axis bridge — MediaPipe → rakija

Single per-joint transform in `terka/joints.py:_vec`:

```python
rakija_x = -mp.z                # MP "into scene"  → rakija forward
rakija_y = -mp.y + 1.10         # MP "down"        → rakija up
                                #                    + lift pelvis to canvas anchor
rakija_z = +mp.x                # MP "image right" → rakija anatomical-left
```

After this, the THETIS subject lands facing rakija's +x with the
pelvis at rakija's canonical y=1.10. r_hip drops to -z (subject's
right side), l_hip to +z, head_center forward of pelvis on +x.
A forehand contact reaches into +x with the racket above shoulder
height (~y=2.0 m).

The combined transform is composed of a y-axis flip (MP image-down
to rakija up), a y-axis lift, and a +90° rotation about +y. All
proper — handedness is preserved, so `pose_from_joints` back-solves
cleanly when "Drive Skeleton from Trail" is on in rakija.

## Racket-tip extrapolation

THETIS has no racket marker, so the dataset alone can't tell us
where the racket head is — only where the wrist is. Stubbing
`r_racket_tip = r_wrist` paints a wrist trail that looks too small
to read as a swing (a forehand "tip" should arc through chest +
overhead, not just chest).

`_racket_tip_from(wrist, elbow)` in `joints.py` extrapolates the
tip along the forearm direction:

```
tip = wrist + 0.55 m × normalise(wrist − elbow)
```

The 55 cm constant is wrist-to-racket-head along the forearm,
based on a standard adult racket geometry. Accurate when the
wrist is locked (forearm + racket roughly collinear); under-rotated
when the wrist breaks (slice / kick serve) — but the swing SHAPE
comes through correctly for the bulk-import flow.

Future kadar capture with an ArUco marker on the racket throat
replaces this guess with a measured pose; the on-disk schema is
unchanged.

## What's still deliberately NOT in the converter

- **Per-clip T-pose calibration.** Our +90° rotation assumes the
  subject is square to the camera. Most THETIS clips are; some
  serves stand sideways and end up rotated weird. Per-clip
  calibration (detect a quiet "ready" frame, derive a basis from
  hips + shoulders) would land each video in a consistent court
  frame — out of scope for the bulk import.
- **Swing-window trim.** Videos are ~4.7 s of setup → swing →
  follow-through. We write the entire video as one trajectory;
  rakija's scrub bar covers the whole thing. Future work: detect
  the contact frame (wrist-velocity peak, same as kadar's
  SwingDetector) and trim to a 1 s window centred on it.

## Filename / metadata convention

THETIS RGB filenames: `p<subject>_<action_short>_s<sequence>.avi`
under `VIDEO_RGB/<canonical_action>/`. We trust the parent
directory for the canonical action label and parse subject +
sequence from the filename.

Mapping by directory:

| dir under `VIDEO_RGB/` | action_short in filenames |
|---|---|
| `backhand` | `backhand` |
| `backhand2hands` | `backhand2h` |
| `backhand_slice` | `bslice` |
| `backhand_volley` | `bvolley` |
| `forehand_flat` | `foreflat` |
| `forehand_openstands` | `foreopen` |
| `forehand_slice` | `fslice` |
| `forehand_volley` | `fvolley` |
| `flat_service` | `serflat` |
| `kick_service` | `serkick` |
| `slice_service` | `serslice` |
| `smash` | `smash` |

Subjects p1 to p31 are beginners; p32 to p55 are experts.
# terka
