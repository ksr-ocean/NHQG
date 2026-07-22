#!/usr/bin/env python
"""Plot global mean-temperature exchange diagnostics from a spectrum archive."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nhqg.paths import normalize_output_dir, resolve_existing_output_path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=str,
        default=(
            "output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_"
            "Nx128_Nz128_dt5e5_t8/spectra/spectrum_history.npz"
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _choose_linthresh(arrays: list[np.ndarray]) -> float:
    finite = np.concatenate([np.abs(a[np.isfinite(a)]) for a in arrays if np.any(np.isfinite(a))])
    if finite.size == 0:
        return 1.0
    p50 = float(np.percentile(finite, 50.0))
    p90 = float(np.percentile(finite, 90.0))
    linthresh = max(1e-300, min(p50, p90 / 10.0 if p90 > 0 else p50))
    if not np.isfinite(linthresh) or linthresh <= 0.0:
        linthresh = 1.0
    return linthresh


def _symlog(values: np.ndarray, linthresh: float) -> np.ndarray:
    return np.sign(values) * np.log10(1.0 + np.abs(values) / linthresh)


def _palette(n: int) -> list[tuple[int, int, int]]:
    base = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
        (23, 190, 207),
    ]
    return [base[i % len(base)] for i in range(n)]


def _draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                times: np.ndarray, series: list[tuple[str, np.ndarray]],
                title: str, mode: str = "symlog"):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(0, 0, 0), width=1)
    font = ImageFont.load_default()
    draw.text((x0 + 6, y0 + 6), title, fill=(0, 0, 0), font=font)

    clean_series = [(name, np.nan_to_num(values, nan=0.0)) for name, values in series]

    if mode == "symlog":
        linthresh = _choose_linthresh([values for _, values in clean_series])
        transformed = [(name, _symlog(values, linthresh)) for name, values in clean_series]
        draw.text((x0 + 180, y0 + 6), f"symlog linthresh={linthresh:.1e}", fill=(80, 80, 80), font=font)
    else:
        transformed = clean_series

    finite_vals = [vals[np.isfinite(vals)] for _, vals in transformed if np.any(np.isfinite(vals))]
    if finite_vals:
        stacked = np.concatenate(finite_vals)
        ymin = float(np.min(stacked))
        ymax = float(np.max(stacked))
    else:
        ymin, ymax = -1.0, 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    left = x0 + 60
    right = x1 - 150
    top = y0 + 28
    bottom = y1 - 30

    if ymin < 0.0 < ymax:
        y_mid = top + int(round((ymax - 0.0) / (ymax - ymin) * (bottom - top)))
        draw.line([(left, y_mid), (right, y_mid)], fill=(170, 170, 170), width=1)
    draw.line([(left, top), (left, bottom)], fill=(0, 0, 0), width=1)
    draw.line([(left, bottom), (right, bottom)], fill=(0, 0, 0), width=1)

    t0 = float(times[0])
    t1 = float(times[-1]) if len(times) > 1 else t0 + 1.0
    span = max(t1 - t0, 1e-12)

    for frac, label in [(0.0, ymax), (0.5, 0.5 * (ymax + ymin)), (1.0, ymin)]:
        y = top + int(frac * (bottom - top))
        draw.text((x0 + 4, y - 6), f"{label:.2e}", fill=(0, 0, 0), font=font)

    for frac in [0.0, 0.5, 1.0]:
        x = left + int(frac * (right - left))
        draw.text((x - 10, bottom + 8), f"{t0 + frac * span:.1f}", fill=(0, 0, 0), font=font)

    colors = _palette(len(series))
    denom = max(ymax - ymin, 1e-12)
    for (name, values), color in zip(transformed, colors):
        pts = []
        for t, yv in zip(times, values):
            px = left + int(round((float(t) - t0) / span * (right - left)))
            py = top + int(round((ymax - float(yv)) / denom * (bottom - top)))
            pts.append((px, py))
        draw.line(pts, fill=color, width=2)

    legend_x = right + 12
    legend_y = top
    for (name, _), color in zip(series, colors):
        draw.rectangle([legend_x, legend_y + 3, legend_x + 12, legend_y + 11], fill=color)
        draw.text((legend_x + 18, legend_y), name, fill=(0, 0, 0), font=font)
        legend_y += 18


def main():
    args = _parse_args()
    archive = resolve_existing_output_path(args.archive)
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")

    out_dir = (
        Path(normalize_output_dir(args.output_dir))
        if args.output_dir
        else archive.parent / "mean_exchange_plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(archive)
    t = np.asarray(data["t"], dtype=np.float64)

    def _optional(name: str) -> np.ndarray:
        if name not in data:
            return np.full_like(t, np.nan, dtype=np.float64)
        return np.asarray(data[name], dtype=np.float64)

    theta_mf_dealiased = _optional("th_mean_feedback_sum_dealiased")
    if np.all(np.isnan(theta_mf_dealiased)) and "th_mean_feedback_shell_tendency_dealiased" in data:
        theta_mf_dealiased = np.sum(
            np.asarray(data["th_mean_feedback_shell_tendency_dealiased"], dtype=np.float64),
            axis=1,
        )

    width, height = 1200, 1240
    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), f"Mean Exchange History: {archive}", fill=(0, 0, 0), font=font)

    _draw_panel(
        draw,
        (20, 50, 1180, 360),
        t,
        [
            ("mean flux exch", _optional("mean_flux_exchange_tendency")),
            ("theta mean fb raw", _optional("th_mean_feedback_sum")),
            ("theta mean fb deal", theta_mf_dealiased),
            ("mean diffusion", _optional("mean_diffusion_tendency")),
            ("exchange raw", _optional("mean_theta_exchange_residual")),
            ("exchange deal", _optional("mean_theta_exchange_residual_dealiased")),
            ("mean total", _optional("mean_total_tendency")),
        ],
        "Mean-Reservoir Budget Terms",
        mode="symlog",
    )
    _draw_panel(
        draw,
        (20, 380, 1180, 690),
        t,
        [
            ("Nu dealiased", _optional("Nusselt_dealiased")),
            ("Nu raw", _optional("Nusselt")),
            ("tw dealiased", _optional("vol_avg_tw_dealiased")),
            ("tw raw", _optional("vol_avg_tw")),
            ("flux mismatch", _optional("heat_flux_mismatch")),
        ],
        "Heat-Flux Diagnostics",
        mode="symlog",
    )
    _draw_panel(
        draw,
        (20, 710, 1180, 1020),
        t,
        [
            ("mean grad min", _optional("mean_grad_min")),
            ("mean grad mid", _optional("mean_grad_mid")),
            ("mean grad max", _optional("mean_grad_max")),
        ],
        "Modified Background Gradient g(z)=1-dTheta_bar/dz",
        mode="linear",
    )
    _draw_panel(
        draw,
        (20, 1040, 1180, 1220),
        t,
        [
            ("th_bar phys max", _optional("th_bar_phys_max")),
            ("dth_bar_dz max", _optional("dth_bar_dz_max")),
            ("mean energy", _optional("mean_energy")),
        ],
        "Mean-Profile Amplitude Diagnostics",
        mode="symlog",
    )

    out_path = out_dir / "mean_exchange_history.png"
    canvas.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
