"""Per-shot-class default strike parameters for THETIS clips.

THETIS recordings have no actual incoming ball or racket-marker
contact — the subject is alone, swinging at thin air. To make
rakija's strike pipeline produce a sensible ball trajectory the
moment a trajectory loads, terka injects a small `strike_params`
block per clip with class-appropriate defaults: incoming ball
(speed, elev, az, spin), contact point (court x/y/z + ball-frame
lon/lat + string-bed u/v), and racket angles (face, swing path
az/elev). Numeric values are calibrated against rakija's existing
GROUNDSTROKE / SERVE_DEUCE / SERVE_AD defaults in
`projects/rakija/strike.c:strike_params_defaults`.

Court coordinate constants (mirrored from rakija/physics.h):

  SIM_X_MAX     = 50.0   m  along court length (player → opponent)
  SIM_Z_MAX     = 25.0   m  across court width
  COURT_LEN     = 23.77  m
  COURT_X_LO    = 13.11  m  (player's baseline)
  COURT_X_HI    = 36.89  m  (opponent's baseline)
  NET_X         = 25.0   m
  COURT_Z_MID   = 12.5   m

Players hit ~2.65 m behind their own baseline for ground strokes,
and ~0.5 m inside it for serves — so default `contact.x_m` for
ground strokes is `COURT_X_LO − 2.65 ≈ 10.46`, and for serves
`COURT_X_LO + 0.5 ≈ 13.61`. The user is expected to edit any of
these in rakija and Save to Vertex if the defaults don't fit.
"""

from __future__ import annotations

from typing import Any

COURT_X_LO  = 13.11
COURT_Z_MID = 12.50


def _ground_stroke(*, face_deg: float = 6.0,
                   path_elev_deg: float = 35.0,
                   in_topspin_rpm: float = 800.0,
                   in_sidespin_rpm: float = 0.0,
                   ball_lat_deg: float | None = -8.0,
                   ) -> dict[str, Any]:
    """Forehand/backhand template — mid-paced rally ball at waist
    height, modest topspin, ~2.65 m behind baseline."""
    return {
        "incoming": {
            "speed_mps":     15.0,
            "elev_deg":      -5.0,
            "az_deg":         0.0,
            "topspin_rpm":   in_topspin_rpm,
            "sidespin_rpm":  in_sidespin_rpm,
        },
        "contact": {
            "x_m":           COURT_X_LO - 2.65,
            "y_m":           0.95,
            "z_m":           COURT_Z_MID,
            "ball_lon_deg":  0.0,
            "ball_lat_deg":  ball_lat_deg,
            "bed_u_mm":      0.0,
            "bed_v_mm":      0.0,
        },
        "racket": {
            "face_angle_deg":      face_deg,
            "swing_path_az_deg":   0.0,
            "swing_path_elev_deg": path_elev_deg,
        },
    }


def _volley(*, face_deg: float = 8.0) -> dict[str, Any]:
    """Volley template — taking the ball earlier (closer to the net,
    forward of own baseline), short swing, less spin."""
    d = _ground_stroke(face_deg=face_deg, path_elev_deg=5.0,
                       in_topspin_rpm=300.0, ball_lat_deg=-3.0)
    d["contact"]["x_m"] = COURT_X_LO + 4.0   # well inside the court
    d["contact"]["y_m"] = 1.20
    return d


def _slice_ground(*, face_deg: float = 14.0) -> dict[str, Any]:
    """Slice ground stroke — slight underspin, open face, flatter
    path. (Negative topspin = underspin in our schema convention.)"""
    return _ground_stroke(
        face_deg=face_deg,
        path_elev_deg=-8.0,
        in_topspin_rpm=-300.0,
        ball_lat_deg=+5.0,
    )


def _serve(*, az_deg: float = +8.0,
           in_sidespin_rpm: float = 0.0,
           in_topspin_rpm:  float = 0.0,
           face_deg: float = 0.0,
           ) -> dict[str, Any]:
    """Serve template — toss apex, contact overhead, hitting down.
    `az_deg` flips between the deuce court (+8°) and ad court (−8°)."""
    return {
        "incoming": {
            "speed_mps":     1.0,
            "elev_deg":     -90.0,
            "az_deg":        0.0,
            "topspin_rpm":  in_topspin_rpm,
            "sidespin_rpm": in_sidespin_rpm,
        },
        "contact": {
            "x_m":           COURT_X_LO + 0.5,
            "y_m":           2.44,
            "z_m":           COURT_Z_MID - 1.5,  # deuce side default
            "ball_lon_deg":  0.0,
            "ball_lat_deg":  0.0,
            "bed_u_mm":      0.0,
            "bed_v_mm":      0.0,
        },
        "racket": {
            "face_angle_deg":      face_deg,
            "swing_path_az_deg":   az_deg,
            "swing_path_elev_deg": -3.0,
        },
    }


# THETIS action directory → strike_params defaults.
DEFAULTS_BY_ACTION: dict[str, dict[str, Any]] = {
    # Right-handed forehand-family — modest topspin, slight closed
    # face, low-to-high path. Open stance lifts the contact_y slightly
    # (more upright bat-end finish); slice/volley tweak spin + path.
    "forehand_flat":         _ground_stroke(),
    "forehand_openstands":   _ground_stroke(),
    "forehand_slice":        _slice_ground(),
    "forehand_volley":       _volley(),

    # Backhand-family — same court geometry as forehand defaults but
    # contact slightly higher (one-hand backhand contact tends to land
    # at ribcage height) and a less-closed face on the slice variant.
    "backhand":              _ground_stroke(face_deg=4.0),
    "backhand2hands":        _ground_stroke(face_deg=5.0),
    "backhand_slice":        _slice_ground(face_deg=12.0),
    "backhand_volley":       _volley(face_deg=6.0),

    # Serves — all default to the deuce-side toss + overhead contact.
    # The kick adds heavy topspin + sidespin, slice trades topspin
    # for sidespin, flat keeps both spins ≈ 0.
    "flat_service":          _serve(),
    "kick_service":          _serve(in_topspin_rpm=1500.0,
                                    in_sidespin_rpm=600.0,
                                    face_deg=4.0),
    "slice_service":         _serve(in_sidespin_rpm=1500.0,
                                    face_deg=6.0),

    # Smash — similar to a serve but with a moderately falling ball
    # (the opponent's lob) rather than a stationary toss.
    "smash":                 {
        "incoming": {
            "speed_mps":     8.0,
            "elev_deg":     -70.0,
            "az_deg":         0.0,
            "topspin_rpm":  200.0,
            "sidespin_rpm":   0.0,
        },
        "contact": {
            "x_m":           COURT_X_LO + 3.0,    # advanced into the court
            "y_m":           2.40,                # overhead
            "z_m":           COURT_Z_MID,
            "ball_lon_deg":  0.0,
            "ball_lat_deg":  0.0,
            "bed_u_mm":      0.0,
            "bed_v_mm":      0.0,
        },
        "racket": {
            "face_angle_deg":       0.0,
            "swing_path_az_deg":    0.0,
            "swing_path_elev_deg": -25.0,
        },
    },
}


def default_for_action(action: str) -> dict[str, Any] | None:
    """Look up the strike-params default block for a THETIS action
    directory name. Returns None for unknown actions so the caller
    can decide whether to fall back to a generic default or omit
    the strike_params block entirely."""
    return DEFAULTS_BY_ACTION.get(action)
