"""terka CLI — convert THETIS videos and ship them to vertex.

Subcommands:
  convert  one video → rakija JSON on stdout (or to a file)
  ingest   walk a THETIS RGB tree, POST each video's trajectory to
           vertex with a thetis:expertise:p<N>:<action>:s<M>
           device_id
  download-model  fetch a Pose Landmarker .task file from Google's
                  CDN so MediaPipe has something to run against
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import click

from contextlib import ExitStack

from terka.classify import (
    TennisShotClassifier,
    build_self_trained_corpus,
    classify_trajectory_json,
    clear_cached_classifier,
    load_self_trained_corpus,
)
from terka.client import VertexClient
from terka.joints import (
    apply_racket_tip,
    smooth_samples,
    trim_to_swing_window,
)
from terka.meta import iter_dataset, parse_video
from terka.pose import (
    detect_ensemble,
    detect_video,
    detect_with_landmarker,
    make_landmarker,
)
from terka.trajectory import to_json_text, trajectory_doc


MODEL_URLS = {
    "lite":  ("https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_lite/float16/latest/"
              "pose_landmarker_lite.task"),
    "full":  ("https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_full/float16/latest/"
              "pose_landmarker_full.task"),
    "heavy": ("https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_heavy/float16/latest/"
              "pose_landmarker_heavy.task"),
}


@click.group()
def cli():
    """terka — THETIS-to-rakija pose-trajectory converter."""


@cli.command()
@click.option("--variant", type=click.Choice(list(MODEL_URLS)),
              default="lite", show_default=True,
              help="Which Pose Landmarker model to grab.")
@click.option("--dest", type=click.Path(path_type=Path),
              default=Path("./pose_landmarker.task"), show_default=True)
def download_model(variant: str, dest: Path):
    """Download a Pose Landmarker .task model from Google's CDN."""
    url = MODEL_URLS[variant]
    click.echo(f"downloading {variant} → {dest}", err=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))
    click.echo(f"saved {dest} ({dest.stat().st_size} bytes)", err=True)


@cli.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("--model", "model_path", type=click.Path(exists=True, path_type=Path),
              help="Path to a single pose_landmarker_*.task. Mutually exclusive with --ensemble.")
@click.option("--ensemble", "ensemble_paths", type=click.Path(exists=True, path_type=Path),
              multiple=True,
              help="Path to a Pose Landmarker model file. Pass multiple times "
                   "(typically --ensemble LITE --ensemble FULL --ensemble HEAVY) "
                   "to run all variants and average per-frame using Varia 2018's "
                   "iterative-mean update. Mutually exclusive with --model.")
@click.option("--pretty/--no-pretty", default=False)
@click.option("--raw/--cleaned", default=False,
              help="--raw skips smoothing + swing-window trim; default --cleaned.")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Write to file instead of stdout.")
def convert(video: Path, model_path: Path | None,
            ensemble_paths: tuple[Path, ...],
            pretty: bool, raw: bool, out: Path | None):
    """Convert one video to a rakija-schema trajectory JSON.

    Default = single model (LITE) at ~1.1 s/video.
    --ensemble = N models averaged per-frame at ~N × that.
    """
    if not model_path and not ensemble_paths:
        raise click.UsageError(
            "pass either --model PATH or one or more --ensemble PATH",
        )
    if model_path and ensemble_paths:
        raise click.UsageError(
            "--model and --ensemble are mutually exclusive",
        )

    meta = parse_video(video)
    extra = None
    if meta is not None:
        extra = {
            "thetis": {
                "subject_num": meta.subject_num,
                "action": meta.action,
                "sequence": meta.sequence,
                "expertise": meta.expertise,
            }
        }

    if ensemble_paths:
        with ExitStack() as stack:
            lms = [stack.enter_context(make_landmarker(p))
                   for p in ensemble_paths]
            samples = list(detect_ensemble(video, lms))
    else:
        samples = list(detect_video(video, model_path))

    if not samples:
        click.echo(f"WARN no frames detected in {video}", err=True)
    if not raw and len(samples) >= 2:
        samples = smooth_samples(samples)
        samples, _ = trim_to_swing_window(samples)
        apply_racket_tip(samples)
    doc = trajectory_doc(samples, extra=extra)
    text = to_json_text(doc, pretty=pretty)
    if out:
        out.write_text(text)
        click.echo(f"wrote {len(samples)} frames → {out}", err=True)
    else:
        sys.stdout.write(text)


@cli.command()
@click.argument("rgb_root", type=click.Path(exists=True, path_type=Path))
@click.option("--model", "model_path", type=click.Path(exists=True, path_type=Path),
              help="Path to a single pose_landmarker_*.task. Mutually exclusive with --ensemble.")
@click.option("--ensemble", "ensemble_paths", type=click.Path(exists=True, path_type=Path),
              multiple=True,
              help="Pass multiple times to run an N-model ensemble averaged "
                   "per-frame (Varia 2018 update). Slower by ~N×.")
@click.option("--vertex-url", default="http://127.0.0.1:8000",
              show_default=True, help="vertex base URL")
@click.option("--limit", type=int, default=None,
              help="Stop after N videos (smoke-test mode).")
@click.option("--actions", "actions_filter", default="",
              help="Comma-separated action directory names to keep "
                   "(e.g. 'forehand_flat,backhand'). Empty = all.")
def ingest(rgb_root: Path, model_path: Path | None,
           ensemble_paths: tuple[Path, ...],
           vertex_url: str, limit: int | None, actions_filter: str):
    """Walk a THETIS VIDEO_RGB tree; convert + POST each video.

    Builds ONE MediaPipe Pose Landmarker for the whole run and
    threads it through every video — avoids the ~300 ms EGL +
    model-load overhead the single-video path would otherwise
    repeat 1980 times. Cumulative `session_offset_ms` keeps the
    per-frame timestamps monotonic across videos, which is what
    MediaPipe's VIDEO running-mode contract requires.

    Pass `--ensemble` multiple times to run an N-variant ensemble
    averaged per-frame using Varia's update — ~N× slower per
    video, but reduces single-model variance (LITE/FULL/HEAVY
    disagree by 6-9 cm median on r_wrist for THETIS clips).
    """
    if not model_path and not ensemble_paths:
        raise click.UsageError(
            "pass either --model PATH or one or more --ensemble PATH",
        )
    if model_path and ensemble_paths:
        raise click.UsageError(
            "--model and --ensemble are mutually exclusive",
        )

    actions = {a for a in actions_filter.split(",") if a}
    client = VertexClient(vertex_url)
    n_ok = n_skip = n_fail = 0
    # Bumped after each video by that video's duration + a small
    # safety gap so consecutive videos' timestamps don't overlap.
    session_offset_ms = 0

    with ExitStack() as stack:
        if ensemble_paths:
            lms = [stack.enter_context(make_landmarker(p))
                   for p in ensemble_paths]
            def run_detect(meta_path, offset):
                return list(detect_ensemble(
                    meta_path, lms, session_offset_ms=offset,
                ))
        else:
            lm = stack.enter_context(make_landmarker(model_path))
            def run_detect(meta_path, offset):
                return list(detect_with_landmarker(
                    meta_path, lm, session_offset_ms=offset,
                ))

        for i, meta in enumerate(iter_dataset(rgb_root)):
            if actions and meta.action not in actions:
                continue
            if limit is not None and (n_ok + n_fail) >= limit:
                break
            click.echo(
                f"[{i:04d}] {meta.path.name} → {meta.device_id}",
                err=True,
            )
            try:
                samples = run_detect(meta.path, session_offset_ms)
            except Exception as exc:
                click.echo(f"  detect failed: {exc}", err=True)
                n_fail += 1
                # Push the offset along anyway by a conservative 10 s
                # so a partially-consumed video doesn't poison the
                # next call's timestamp expectation.
                session_offset_ms += 10_000
                continue
            # Even on a 0-1 frame video, bump the offset by enough
            # to keep timestamps strictly increasing for the next
            # detect_for_video call.
            video_span_ms = int(samples[-1][0] * 1000) if samples else 0
            session_offset_ms += video_span_ms + 100
            if len(samples) < 2:
                click.echo(
                    f"  skip — only {len(samples)} usable frames",
                    err=True,
                )
                n_skip += 1
                continue
            # Cleanup pass: kill MP LITE jitter + trim to the swing
            # window so the trail reads as a swing instead of 4 s of
            # static-frame noise. See joints.py module docstring.
            samples = smooth_samples(samples)
            samples, _ = trim_to_swing_window(samples)
            apply_racket_tip(samples)
            doc = trajectory_doc(samples, extra={
                "thetis": {
                    "subject_num": meta.subject_num,
                    "action": meta.action,
                    "sequence": meta.sequence,
                    "expertise": meta.expertise,
                }
            })
            try:
                resp = client.upload(doc, device_id=meta.device_id)
            except Exception as exc:
                click.echo(f"  upload failed: {exc}", err=True)
                n_fail += 1
                continue
            click.echo(
                f"  uploaded id={resp.get('id')} "
                f"frames={len(samples)} duration={doc['duration_s']:.2f}s",
                err=True,
            )
            n_ok += 1
    click.echo(
        f"\ndone — ok={n_ok} skip={n_skip} fail={n_fail}", err=True,
    )


@cli.command("build-classifier")
@click.argument("rgb_root", type=click.Path(exists=True, path_type=Path))
@click.option("--model", "model_path", type=click.Path(exists=True, path_type=Path),
              required=True, help="Path to pose_landmarker_*.task")
def build_classifier(rgb_root: Path, model_path: Path):
    """Walk a THETIS RGB tree, build a self-trained classifier corpus.

    For every clip: detect with MediaPipe, smooth + trim to the swing
    window, compute the 225-dim Activity Feature Vector, label by
    THETIS action directory. Write the corpus to
    `~/.cache/terka/corpus.npz` (~40 min on LITE) and a fitted
    classifier to `~/.cache/terka/classifier.pkl`. Future `terka
    classify` calls use this self-consistent classifier in preference
    to Varia's published .mat files.

    Reports the 5-fold cross-validated accuracy on the built corpus
    at the end so you can decide whether the model is worth shipping.
    """
    import numpy as np
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    corpus = build_self_trained_corpus(
        rgb_root, model_path, progress=lambda m: click.echo(m, err=True),
    )
    click.echo(
        f"\ncorpus stats: {corpus.X.shape[0]} swings × {corpus.X.shape[1]} "
        f"features, {len(np.unique(corpus.y))} classes",
        err=True,
    )
    # Fit + persist
    clear_cached_classifier()
    cls = TennisShotClassifier().fit(corpus)
    cache_dir = Path.home() / ".cache" / "terka"
    cache_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(cache_dir / "classifier.pkl", "wb") as f:
        pickle.dump(cls, f)
    click.echo(f"wrote classifier → {cache_dir / 'classifier.pkl'}", err=True)

    # Honest cross-validated accuracy on the just-built corpus.
    # Auto-shrink n_splits so a sparse smoke-test corpus doesn't crash.
    # Use Counter, not np.bincount — bincount returns counts indexed
    # from 0, but our labels start at 1 so index-0 is always empty,
    # which would falsely report "smallest class has 0 samples".
    from collections import Counter
    smallest_class = min(Counter(corpus.y.tolist()).values())
    n_splits = min(5, smallest_class)
    if n_splits < 2:
        click.echo(
            f"smallest class has {smallest_class} sample(s) — "
            "skipping cross-validation",
            err=True,
        )
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        fresh = TennisShotClassifier()
        scores = cross_val_score(fresh.pipeline, corpus.X, corpus.y, cv=cv)
        click.echo(
            f"{n_splits}-fold CV accuracy: "
            f"{scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%",
            err=True,
        )


@cli.command()
@click.argument("trajectory", type=click.Path(exists=True, path_type=Path))
@click.option("--pretty/--no-pretty", default=True)
def classify(trajectory: Path, pretty: bool):
    """Classify a rakija-format trajectory JSON by shot class.

    Uses Varia 2018's Activity-Feature-Vector pipeline (torso-
    normalised joint distances → k-means representative postures →
    multi-class SVM) trained on the 681 labelled swings shipped
    in `ref/thetis-dataset-improvement/.../input_results/*.mat`.
    Outputs the predicted action + decision-function score per
    class.
    """
    result = classify_trajectory_json(trajectory)
    indent = 2 if pretty else None
    click.echo(json.dumps(result, indent=indent))


if __name__ == "__main__":
    cli()
