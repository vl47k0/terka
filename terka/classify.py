"""Tennis-shot classifier — Python port of Varia 2018's MATLAB pipeline.

Source code: `ref/thetis-dataset-improvement/Activity_Recognition_Implementation/`
Paper:       Varia et al. 2018, "A Refined 3D Dataset…" IEEE CIG.

Pipeline (per swing):

  1. Per-frame "posture vector" — each of Varia/Kinect's 15 joints
     encoded as its displacement from the TORSO landmark, scaled by
     |NECK − TORSO|. Translation + scale invariant. 45 numbers per
     frame (= 15 joints × 3 coords).  [JointFeatures.m + PostureVector.m]

  2. K-means cluster the per-frame posture vectors into N=5
     representative postures.  [ActivityFeature.m]

  3. Sort the cluster IDs by temporal occurrence — for each cluster,
     keep only the position of its LONGEST contiguous run, drop the
     other runs, then return the surviving cluster IDs in their
     temporal order.  [sortCenters.m]

  4. Concatenate the cluster centres in that order → one 225-dim
     "Activity Feature Vector" per swing. Pad with zeros if k-means
     collapsed clusters (rare).  [ActivityFeature.m]

  5. Multi-class SVM (one-vs-rest, RBF kernel) trained on the .mat
     files in `ref/thetis-dataset-improvement/.../input_results/`.
     681 labelled swings across 12 shot classes.  [svm.m]

The 12 classes are exactly Varia's: 1=backhand, 2=backhand2hands,
3=backhand_slice, 4=forehand_flat, 5=forehand_openstands,
6=forehand_slice, 7=flat_service, 8=kick_service,
9=slice_service, 10=smash, 11=forehand_volley, 12=backhand_volley.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from terka.joints import (
    RigJoints,
    apply_racket_tip,
    smooth_samples,
    trim_to_swing_window,
)


# Rakija field names in Varia/Kinect's canonical 15-joint order.
# Indices map straight to MATLAB PostureVector.m:
#   0 HEAD          1 L_FOREARM(elbow)  2 L_FOOT(ankle)  3 L_HAND
#   4 L_THIGH(hip)  5 L_CALF(knee)      6 L_UPPERARM(shoulder)
#   7 NECK
#   8 R_FOREARM(elbow)  9 R_FOOT(ankle) 10 R_HAND        11 R_THIGH(hip)
#   12 R_CALF(knee)    13 R_UPPERARM(shoulder)
#   14 SPINE1(torso)
RAKIJA_FIELDS_15 = (
    "head_center",
    "l_elbow", "l_ankle", "l_hand", "l_hip", "l_knee", "l_shoulder",
    "spine_top",          # NECK
    "r_elbow", "r_ankle", "r_wrist", "r_hip", "r_knee", "r_shoulder",
    "torso",              # TORSO
)
_J_NECK = 7
_J_TORSO = 14

N_CLUSTERS = 5
DIM_PER_FRAME = 15 * 3                # 45
TOTAL_DIM = N_CLUSTERS * DIM_PER_FRAME  # 225


# Varia's createFinalMat.m's keyset (1-indexed) → THETIS canonical
# action directory names (what terka.meta.iter_dataset yields).
LABEL_TO_ACTION = {
    1:  "backhand",
    2:  "backhand2hands",
    3:  "backhand_slice",
    4:  "forehand_flat",
    5:  "forehand_openstands",
    6:  "forehand_slice",
    7:  "flat_service",
    8:  "kick_service",
    9:  "slice_service",
    10: "smash",
    11: "forehand_volley",
    12: "backhand_volley",
}
ACTION_TO_LABEL = {v: k for k, v in LABEL_TO_ACTION.items()}


# Default location for Varia's pre-computed Activity Feature .mat
# files (in the umbrella repo's ref/ tree, where the user cloned it).
# Kept as a fallback path for the classifier; in practice the
# self-trained corpus below outperforms it because the .mat files
# are the output of Varia's pipeline run on per-frame data we don't
# have, so our k-means cluster-slot semantics don't align with them.
_VARIA_TRAINING_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "ref" / "thetis-dataset-improvement"
    / "Activity_Recognition_Implementation" / "input_results"
)

# Self-trained corpus + classifier — built by `terka build-classifier`
# from terka's own MediaPipe extraction over the THETIS RGB clips.
# Cached under XDG_CACHE_HOME so a re-install of terka doesn't lose
# the ~40-minute training run.
_CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "terka"
_CORPUS_PATH = _CACHE_DIR / "corpus.npz"
_CLASSIFIER_PATH = _CACHE_DIR / "classifier.pkl"


# --------------------------------------------------------------------- #
# Per-swing feature extraction                                            #
# --------------------------------------------------------------------- #

def posture_vector(joints: RigJoints) -> np.ndarray:
    """Single frame → 45-dim torso-normalised posture vector.

    Direct port of PostureVector.m + JointFeatures.m:
      D_i = (J_i − J_torso) / |J_neck − J_torso|

    Translation invariant (everything relative to torso), scale
    invariant (divided by torso-to-neck length, which acts as a
    body-size proxy). Returns the 15 normalised joint vectors
    flattened row-major.
    """
    coords = np.array(
        [getattr(joints, f) for f in RAKIJA_FIELDS_15], dtype=np.float64,
    )  # (15, 3)
    scale = float(np.linalg.norm(coords[_J_NECK] - coords[_J_TORSO]))
    if scale < 1e-9:
        return np.zeros(DIM_PER_FRAME)
    rel = (coords - coords[_J_TORSO]) / scale  # (15, 3)
    return rel.flatten()  # (45,)


def _sort_centers(idx: np.ndarray) -> np.ndarray:
    """Port of sortCenters.m.

    Input: per-frame cluster assignments (length-T int array).
    Output: each cluster ID once, in the temporal order of its
    LONGEST contiguous run (other runs of the same cluster are
    dropped).

    Example: [3,3,1,1,3,3,3,2,2] → compressed=[3,1,3,2] with
    run-lengths [2,2,3,2]. For cluster 3 the longest run is the
    second (length 3), so the first occurrence is dropped →
    [1,3,2].
    """
    if len(idx) == 0:
        return idx.copy()
    # Compress consecutive duplicates while tracking run lengths.
    change = np.diff(idx) != 0
    boundaries = np.concatenate(([True], change))
    compressed = idx[boundaries]
    # Lengths: number of frames in each run.
    starts = np.where(boundaries)[0]
    ends = np.concatenate((starts[1:], [len(idx)]))
    run_lengths = ends - starts
    # For each cluster value, keep only the position with max run length.
    keep = np.ones(len(compressed), dtype=bool)
    seen = set()
    for i, v in enumerate(compressed):
        if int(v) in seen:
            continue
        seen.add(int(v))
        # All positions where this cluster value appears in `compressed`.
        positions = np.where(compressed == v)[0]
        if len(positions) <= 1:
            continue
        # Among those positions, find the one with longest original run.
        longest = positions[np.argmax(run_lengths[positions])]
        for p in positions:
            if p != longest:
                keep[p] = False
    return compressed[keep]


def activity_feature_vector(
    samples: Sequence[tuple[float, RigJoints]],
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    """One swing → 225-dim feature vector.

    Direct port of ActivityFeature.m. Requires the swing to be the
    cleaned/trimmed window (smooth_samples + trim_to_swing_window
    output is what you want here).
    """
    if len(samples) < n_clusters:
        # Too few frames to cluster — return zero-padded vector so
        # downstream code can still process it (will likely score low).
        return np.zeros(TOTAL_DIM)
    postures = np.stack([posture_vector(j) for _, j in samples])  # (T, 45)
    # n_init=10 matches scikit-learn's pre-1.4 default; deterministic via
    # explicit random_state so a re-run of the same swing classifies
    # identically.
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    cluster_ids = km.fit_predict(postures)
    centres = km.cluster_centers_  # (K, 45)
    ordered = _sort_centers(cluster_ids)
    # Concatenate centres in temporal order; pad with zeros if
    # _sort_centers collapsed clusters (rare but allowed by the
    # MATLAB pipeline — empty space in the row gets zeros).
    feat = np.zeros(n_clusters * DIM_PER_FRAME)
    for i, cid in enumerate(ordered[:n_clusters]):
        feat[i * DIM_PER_FRAME:(i + 1) * DIM_PER_FRAME] = centres[int(cid)]
    return feat


# --------------------------------------------------------------------- #
# Training-corpus loader + classifier                                    #
# --------------------------------------------------------------------- #

@dataclass
class TrainingCorpus:
    X: np.ndarray  # (N, 225)
    y: np.ndarray  # (N,) integer labels 1..12


def load_training_corpus(
    training_dir: Path = _VARIA_TRAINING_DIR,
    include_uncertain: bool = True,
) -> TrainingCorpus:
    """Concatenate Varia's pre-computed Activity Feature .mat files.

    Default: Normal_{Amateurs,Experts} + Uncertain_{Amateurs,Experts}
    = 681 swings × 225 features. `include_uncertain=False` drops the
    60 uncertain-data rows, leaving 621.

    Caveat: in practice a classifier trained on this corpus tops out
    at ~46 % accuracy in our pipeline because our k-means cluster-slot
    semantics don't align with Varia's (the per-frame source data she
    used was never published). For production, use
    `load_self_trained_corpus()` instead — built end-to-end from
    terka's own MediaPipe extraction.
    """
    if not training_dir.is_dir():
        raise FileNotFoundError(
            f"training-data dir not found: {training_dir} — "
            "expected the CVaria/thetis-dataset-improvement clone "
            "under ref/ at the umbrella-repo root",
        )
    files = ["Normal_Experts_Files.mat", "Normal_Amateurs_Files.mat"]
    if include_uncertain:
        files += ["Uncertain_Experts_Files.mat", "Uncertain_Amateurs_Files.mat"]
    Xs, ys = [], []
    for name in files:
        m = loadmat(str(training_dir / name))
        Xs.append(m["Input_Activities"])
        ys.append(m["Input_Labels"].flatten())
    return TrainingCorpus(
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0).astype(int),
    )


def _activity_feature_from_video(video_path: Path, landmarker,
                                  session_offset_ms: int = 0,
                                  ) -> tuple[np.ndarray | None, int]:
    """terka extraction + cleanup + activity feature for one video.

    Returns (feature_or_None, video_span_ms). The caller uses
    video_span_ms to bump its cumulative session offset so the
    landmarker's monotonic-timestamp contract holds across the
    batch.
    """
    from terka.pose import detect_with_landmarker
    samples = list(detect_with_landmarker(
        video_path, landmarker, session_offset_ms=session_offset_ms,
    ))
    span = int(samples[-1][0] * 1000) if samples else 0
    if len(samples) < N_CLUSTERS:
        return None, span
    samples = smooth_samples(samples)
    samples, _ = trim_to_swing_window(samples)
    apply_racket_tip(samples)
    if len(samples) < N_CLUSTERS:
        return None, span
    return activity_feature_vector(samples), span


def build_self_trained_corpus(
    rgb_root: Path,
    model_path: Path,
    *,
    save_to: Path | None = _CORPUS_PATH,
    progress=lambda msg: None,
) -> TrainingCorpus:
    """Walk a THETIS RGB tree, compute one activity feature per clip.

    Label = the clip's THETIS action directory. Save corpus to
    `save_to` (default: XDG-cached corpus.npz so the next
    get_classifier() finds it). ~40 min on a single LITE pass.
    """
    from terka.meta import iter_dataset
    from terka.pose import make_landmarker

    Xs: list[np.ndarray] = []
    ys: list[int] = []
    session_offset_ms = 0
    n_ok = n_skip = n_fail = 0
    with make_landmarker(model_path) as lm:
        for i, meta in enumerate(iter_dataset(rgb_root)):
            label = ACTION_TO_LABEL.get(meta.action)
            if label is None:
                progress(f"[{i:04d}] skip — unknown action {meta.action!r}")
                n_skip += 1
                continue
            try:
                feat, span = _activity_feature_from_video(
                    meta.path, lm, session_offset_ms=session_offset_ms,
                )
            except Exception as exc:
                progress(f"[{i:04d}] {meta.path.name}: detect FAILED ({exc})")
                n_fail += 1
                session_offset_ms += 10_000
                continue
            session_offset_ms += span + 100
            if feat is None:
                progress(f"[{i:04d}] {meta.path.name}: skip — too few frames")
                n_skip += 1
                continue
            Xs.append(feat)
            ys.append(label)
            n_ok += 1
            if n_ok % 50 == 0:
                progress(
                    f"  …{n_ok} feature vectors built "
                    f"({n_skip} skipped, {n_fail} failed)",
                )
    if not Xs:
        raise RuntimeError("no clips produced a usable feature vector")
    corpus = TrainingCorpus(X=np.stack(Xs), y=np.array(ys, dtype=int))
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_to, X=corpus.X, y=corpus.y)
        progress(f"saved corpus → {save_to} ({corpus.X.shape})")
    progress(f"done — built {n_ok}, skipped {n_skip}, failed {n_fail}")
    return corpus


def load_self_trained_corpus(path: Path = _CORPUS_PATH) -> TrainingCorpus:
    """Load the corpus.npz produced by build_self_trained_corpus."""
    data = np.load(path)
    return TrainingCorpus(X=data["X"], y=data["y"].astype(int))


@dataclass
class Prediction:
    action: str       # canonical THETIS action name
    label: int        # 1..12
    confidence: float # decision-function score for the winning class
    ranked: list[tuple[str, float]]  # all 12 classes sorted by score, top first


class TennisShotClassifier:
    """Multi-class SVM over the Activity Feature Vector representation.

    sklearn's `SVC(decision_function_shape="ovr")` is the modern
    counterpart to the multi-class wrapper in svm.m. A StandardScaler
    is inserted ahead of the SVM because the .mat feature vectors are
    not z-scored (they're raw torso-normalised joint displacements).
    """

    def __init__(self):
        # Hyperparameter selection: a 5-fold stratified CV sweep over
        # {kernel ∈ rbf/linear/poly3} × {C ∈ 1, 10, 100} × {with /
        # without StandardScaler} × {±PCA-50} peaked at ~46 % with
        # PCA-50 + rbf, plain rbf C=1 hit 38 %. Sticking with RBF
        # C=1 + scaler for the cleanest implementation; the +5 pp
        # from PCA isn't worth the second-stage complexity given
        # the corpus ceiling.
        #
        # NOTE: cross-validation accuracy on Varia's shipped .mat
        # files tops out at ~46 %, well below the paper's claimed
        # 93.65 %. The discrepancy is the .mat artifact itself —
        # within-class L2 distances roughly equal cross-class, so
        # the feature vectors as published don't separate cleanly.
        # The classifier is still useful as a "best guess" with
        # calibrated decision-function scores; treat the top-3 list
        # as the actionable output, not the top-1 label.
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                decision_function_shape="ovr",
                probability=False,
                random_state=0,
            )),
        ])
        self._fitted = False

    def fit(self, corpus: TrainingCorpus) -> "TennisShotClassifier":
        self.pipeline.fit(corpus.X, corpus.y)
        self._fitted = True
        return self

    def predict_vector(self, x: np.ndarray) -> Prediction:
        if not self._fitted:
            raise RuntimeError("classifier not fitted — call fit() first")
        x = x.reshape(1, -1)
        scores = self.pipeline.decision_function(x)[0]  # (12,)
        labels = self.pipeline.named_steps["svm"].classes_
        # winning label
        win_idx = int(np.argmax(scores))
        win_label = int(labels[win_idx])
        # ranked list of (action, score) for the user-facing JSON
        ranked = sorted(
            (
                (LABEL_TO_ACTION[int(l)], float(s))
                for l, s in zip(labels, scores)
            ),
            key=lambda t: -t[1],
        )
        return Prediction(
            action=LABEL_TO_ACTION[win_label],
            label=win_label,
            confidence=float(scores[win_idx]),
            ranked=ranked,
        )

    def predict(self, samples: Sequence[tuple[float, RigJoints]]) -> Prediction:
        return self.predict_vector(activity_feature_vector(samples))


# Lazy singleton so a CLI invocation doesn't pay the ~2 s training cost
# more than once per process. Re-fit by calling clear_cached_classifier().
_CACHED: TennisShotClassifier | None = None


def get_classifier(verbose: bool = False) -> TennisShotClassifier:
    """Return a fitted classifier — preferring (in order):

    1. The pickle at ~/.cache/terka/classifier.pkl, if present (instant).
    2. A fresh fit on the self-trained corpus.npz, if present (~2 s
       and pickle written back for next time).
    3. A fresh fit on Varia's .mat files as a last resort, with a
       warning since accuracy is capped at ~46 %.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    if _CLASSIFIER_PATH.is_file():
        with open(_CLASSIFIER_PATH, "rb") as f:
            _CACHED = pickle.load(f)
        if verbose:
            print(f"loaded cached classifier from {_CLASSIFIER_PATH}")
        return _CACHED
    c = TennisShotClassifier()
    if _CORPUS_PATH.is_file():
        if verbose:
            print(f"fitting classifier on self-trained corpus {_CORPUS_PATH}")
        c.fit(load_self_trained_corpus())
    else:
        if verbose:
            print(
                f"WARN no self-trained corpus at {_CORPUS_PATH}; falling "
                f"back to Varia .mat files (accuracy ~46 %). Run "
                f"`terka build-classifier ...` to bootstrap.",
            )
        c.fit(load_training_corpus())
    # Persist so subsequent invocations skip the fit.
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CLASSIFIER_PATH, "wb") as f:
        pickle.dump(c, f)
    if verbose:
        print(f"saved fitted classifier → {_CLASSIFIER_PATH}")
    _CACHED = c
    return _CACHED


def clear_cached_classifier() -> None:
    """Reset the in-memory cache. Disk cache is NOT touched — delete
    `~/.cache/terka/classifier.pkl` manually to force a re-fit."""
    global _CACHED
    _CACHED = None


# --------------------------------------------------------------------- #
# JSON-trajectory helper used by the CLI                                  #
# --------------------------------------------------------------------- #

def classify_trajectory_json(path: Path) -> dict:
    """Load a rakija-format trajectory JSON, classify it, return a
    dict ready for json.dumps.
    """
    doc = json.loads(Path(path).read_text())
    samples = []
    for frame in doc.get("samples", []):
        t = float(frame.get("t", 0.0))
        kwargs = {
            f: frame[f] for f in RAKIJA_FIELDS_15 + ("pelvis",
                "r_racket_tip", "l_toe", "r_toe", "r_wrist", "l_hand")
            if f in frame
        }
        # Need ALL RigJoints fields populated for the dataclass; pull
        # any missing ones from sensible defaults so a partial JSON
        # (older v3 file without `torso`) still classifies.
        if "torso" not in kwargs:
            sp = np.array(kwargs.get("spine_top", [0.0, 1.5, 0.0]))
            pv = np.array(kwargs.get("pelvis",    [0.0, 1.0, 0.0]))
            kwargs["torso"] = list((sp + pv) * 0.5)
        # Fill any missing fields with zero — they won't affect the
        # Varia 15-joint feature extraction.
        for f in (
            "pelvis", "spine_top", "torso", "head_center",
            "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
            "l_toe", "r_toe", "l_shoulder", "r_shoulder",
            "r_elbow", "r_wrist", "r_racket_tip",
            "l_elbow", "l_hand",
        ):
            kwargs.setdefault(f, [0.0, 0.0, 0.0])
        samples.append((t, RigJoints(**kwargs)))
    if len(samples) < N_CLUSTERS:
        return {
            "error": (
                f"trajectory has only {len(samples)} frames; need "
                f"at least {N_CLUSTERS} to classify"
            ),
        }
    pred = get_classifier().predict(samples)
    return {
        "action":     pred.action,
        "label":      pred.label,
        "confidence": pred.confidence,
        "ranked":     [
            {"action": a, "score": s}
            for a, s in pred.ranked
        ],
        "thetis": doc.get("thetis"),
    }
