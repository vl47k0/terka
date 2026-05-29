"""RigJoints → rakija JSON trajectory.

Same format kadar + rakija agreed on:

  {
    "version":    1,
    "duration_s": float,
    "samples": [
      { "t": 0.0, "pelvis": [x,y,z], "spine_top": [x,y,z], ... },
      ...
    ]
  }

Joint field names match rakija's PoseJoints struct field names
exactly so the loader reads them back unmodified.
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
) -> dict:
    """Build a serialisable rakija trajectory document.

    `samples` is the iterable detect_video yields — pairs of
    (t_seconds, joints). `extra` lets the caller inject auxiliary
    fields (subject, action, sequence …) that vertex's POST
    handler stores in payload but rakija's loader ignores.
    """
    if not samples:
        duration = 0.0
    else:
        duration = max(s[0] for s in samples) - min(s[0] for s in samples)
    out: dict = {
        "version": 1,
        "duration_s": float(duration),
        "samples": [
            {"t": float(t), **asdict(joints)}
            for t, joints in samples
        ],
    }
    if extra:
        # Top-level extras — won't disturb the rakija loader (it
        # looks up "samples" + "duration_s" only) and vertex stores
        # the whole dict in payload as a JSONField.
        out.update(extra)
    return out


def to_json_text(doc: dict, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(doc, indent=2)
    return json.dumps(doc, separators=(",", ":"))
