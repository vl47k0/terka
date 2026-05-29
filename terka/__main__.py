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

import sys
import urllib.request
from pathlib import Path

import click

from terka.client import VertexClient
from terka.meta import iter_dataset, parse_video
from terka.pose import detect_video
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
              required=True, help="Path to pose_landmarker_*.task")
@click.option("--pretty/--no-pretty", default=False)
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="Write to file instead of stdout.")
def convert(video: Path, model_path: Path, pretty: bool, out: Path | None):
    """Convert one video to a rakija-schema trajectory JSON."""
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
    samples = list(detect_video(video, model_path))
    if not samples:
        click.echo(f"WARN no frames detected in {video}", err=True)
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
              required=True)
@click.option("--vertex-url", default="http://127.0.0.1:8000",
              show_default=True, help="vertex base URL")
@click.option("--limit", type=int, default=None,
              help="Stop after N videos (smoke-test mode).")
@click.option("--actions", "actions_filter", default="",
              help="Comma-separated action directory names to keep "
                   "(e.g. 'forehand_flat,backhand'). Empty = all.")
def ingest(rgb_root: Path, model_path: Path, vertex_url: str,
           limit: int | None, actions_filter: str):
    """Walk a THETIS VIDEO_RGB tree; convert + POST each video."""
    actions = {a for a in actions_filter.split(",") if a}
    client = VertexClient(vertex_url)
    n_ok = n_skip = n_fail = 0
    for i, meta in enumerate(iter_dataset(rgb_root)):
        if actions and meta.action not in actions:
            continue
        if limit is not None and (n_ok + n_fail) >= limit:
            break
        click.echo(f"[{i:04d}] {meta.path.name} → {meta.device_id}", err=True)
        try:
            samples = list(detect_video(meta.path, model_path))
        except Exception as exc:
            click.echo(f"  detect failed: {exc}", err=True)
            n_fail += 1
            continue
        if len(samples) < 2:
            click.echo(f"  skip — only {len(samples)} usable frames", err=True)
            n_skip += 1
            continue
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


if __name__ == "__main__":
    cli()
