"""THETIS filename parser.

Filenames follow `p<subject>_<action_short>_s<sequence>.avi` —
the parent directory carries the canonical action name (the long
form used in the dataset's directory tree). We trust the parent
dir for the action label rather than try to round-trip
`action_short` → canonical, because the short forms aren't 1:1
(e.g. `backhand2hands` overlaps `backhand` + a suffix).

Subjects p1-p31 are beginners, p32-p55 are experts per the
dataset's README; we tag that as `expertise`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Canonical action names — these match the THETIS directory names
# under VIDEO_RGB/. The label rakija never sees directly, but the
# vertex Trajectory.device_id encodes it so the admin filter +
# rakija "Load from Vertex" can be wired up to scope by action later.
ACTIONS = {
    "backhand", "backhand2hands", "backhand_slice", "backhand_volley",
    "forehand_flat", "forehand_openstands", "forehand_slice",
    "forehand_volley", "flat_service", "kick_service", "slice_service",
    "smash",
}

_SUBJECT_PATTERN = re.compile(r"p(\d+)")
_SEQUENCE_PATTERN = re.compile(r"s(\d+)")


@dataclass(frozen=True)
class VideoMeta:
    """Parsed metadata for one THETIS RGB video."""
    path: Path
    subject_num: int        # 1..55
    action: str             # canonical, matches dataset dir
    sequence: int           # 1..N, per-subject repeat index

    @property
    def expertise(self) -> str:
        return "expert" if self.subject_num >= 32 else "beginner"

    @property
    def device_id(self) -> str:
        """Vertex device_id encoding the full provenance.

        Format `thetis:<expertise>:p<N>:<action>:s<M>` so the admin
        list-filter on device_id can substring-match any of the
        four axes (expertise / subject / action / sequence)
        without needing extra Trajectory columns.
        """
        return (
            f"thetis:{self.expertise}:p{self.subject_num}:"
            f"{self.action}:s{self.sequence}"
        )


def parse_video(path: Path) -> VideoMeta | None:
    """Derive metadata from a THETIS RGB video path.

    Returns None when the filename doesn't match the THETIS
    convention or the action directory isn't one we know.
    """
    if path.suffix.lower() != ".avi":
        return None
    action = path.parent.name
    if action not in ACTIONS:
        return None

    name = path.stem
    sm = _SUBJECT_PATTERN.search(name)
    qm = _SEQUENCE_PATTERN.search(name)
    if not sm or not qm:
        return None
    return VideoMeta(
        path=path,
        subject_num=int(sm.group(1)),
        action=action,
        sequence=int(qm.group(1)),
    )


def iter_dataset(rgb_root: Path):
    """Yield VideoMeta for every parseable .avi under rgb_root.

    Sorted by (action, subject, sequence) so reruns process the
    same dataset in the same order — handy when the run is
    interrupted partway through.
    """
    files = sorted(rgb_root.rglob("*.avi"))
    for f in files:
        m = parse_video(f)
        if m is not None:
            yield m
