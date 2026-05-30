"""RigJoints → rakija JSON trajectory.

Same format kadar + rakija agreed on, bumped to schema **v2** so the
JSON can carry the strike inputs the captured swing doesn't provide
(incoming ball, racket-frame contact, racket face / path):

  {
    "version":      2,
    "duration_s":   float,
    "samples":      [{ "t", "pelvis": [x,y,z], ... }, ...],
    "thetis":       { subject, action, sequence, expertise } (optional),
    "strike_params": {
      "incoming": { "speed_mps", "elev_deg", "az_deg",
                    "topspin_rpm", "sidespin_rpm" },
      "contact":  { "x_m", "y_m", "z_m",
                    "ball_lon_deg", "ball_lat_deg",
                    "bed_u_mm", "bed_v_mm" },
      "racket":   { "face_angle_deg",
                    "swing_path_az_deg",
                    "swing_path_elev_deg" }
    }
  }

Joint field names match rakija's PoseJoints struct field names
exactly so the loader reads them back unmodified.

Schema-v1 payloads (no strike_params) keep loading fine in rakija —
the loader fills missing fields from class-based defaults baked into
the body panel. v2 is forward-compatible: rakija can read both, and
vertex just stores the dict in payload as-is.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Sequence

from terka.joints import RigJoints


def trajectory_doc(
    samples: Sequence[tuple[float, RigJoints]],
    *,
    extra: dict | None = None,
    strike_params: dict | None = None,
) -> dict:
    """Build a serialisable rakija trajectory document (schema v2).

    `samples` is the iterable detect_video yields — pairs of
    (t_seconds, joints). `extra` lets the caller inject auxiliary
    fields (subject, action, sequence …) that vertex's POST
    handler stores in payload but rakija's loader ignores.
    `strike_params` is the optional dict of incoming/contact/racket
    defaults from terka.strike_defaults; rakija loads these into
    its strike spinboxes so the court trajectory is populated as
    soon as the trail lands.
    """
    if not samples:
        duration = 0.0
    else:
        duration = max(s[0] for s in samples) - min(s[0] for s in samples)
    out: dict = {
        "version": 2,
        "duration_s": float(duration),
        "samples": [
            {"t": float(t), **asdict(joints)}
            for t, joints in samples
        ],
    }
    if strike_params:
        out["strike_params"] = strike_params
    if extra:
        # Top-level extras — won't disturb the rakija loader (it
        # looks up "samples" + "duration_s" + "strike_params" only)
        # and vertex stores the whole dict in payload as a JSONField.
        out.update(extra)
    return out


def to_json_text(doc: dict, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(doc, indent=2)
    return json.dumps(doc, separators=(",", ":"))
