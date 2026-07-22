#!/usr/bin/env python
"""Plot shellwise KE-budget diagnostics from a saved spectrum_history archive."""

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
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0],
        help="Simulation times to use for lineout plots.",
    )
    return parser.parse_args()


def _finite_abs_percentile(arr: np.ndarray, q: float) -> float:
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return 1.0
    return float(np.percentile(finite, q))


def _diverging_rgb_unit(scaled: np.ndarray) -> np.ndarray:
    """Map values in [-1, 1] to a blue-white-red palette."""
    scaled = np.clip(scaled, -1.0, 1.0)
    rgb = np.empty(scaled.shape + (3,), dtype=np.uint8)
    pos = scaled >= 0.0
    neg = ~pos
    mag = np.abs(scaled)

    rgb[..., 0] = 255
    rgb[..., 1] = (255 * (1.0 - mag)).astype(np.uint8)
    rgb[..., 2] = (255 * (1.0 - mag)).astype(np.uint8)

    rgb[neg, 0] = (255 * (1.0 - mag[neg])).astype(np.uint8)
    rgb[neg, 1] = (255 * (1.0 - mag[neg])).astype(np.uint8)
    rgb[neg, 2] = 255
    return rgb


def _symlog_transform(data: np.ndarray, linthresh: float) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    return np.sign(data) * np.log10(1.0 + np.abs(data) / linthresh)


def _choose_linthresh(data: np.ndarray) -> float:
    p50 = _finite_abs_percentile(data, 50.0)
    p90 = _finite_abs_percentile(data, 90.0)
    linthresh = max(1e-300, min(p50, p90 / 10.0 if p90 > 0 else p50))
    if not np.isfinite(linthresh) or linthresh <= 0.0:
        linthresh = 1.0
    return linthresh


def _signed_heatmap(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    linthresh = _choose_linthresh(data)
    transformed = _symlog_transform(data, linthresh)
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.max(np.abs(transformed[np.isfinite(transformed)]))) if np.any(np.isfinite(transformed)) else 1.0
    vmax = max(vmax, 1.0)
    rgb = _diverging_rgb_unit(transformed / vmax)
    return rgb, linthresh, vmax


def _draw_heatmap(data: np.ndarray, x_values: np.ndarray, t_values: np.ndarray,
                  out_path: Path, title: str, xlabel: str):
    rgb, linthresh, vmax = _signed_heatmap(data)
    tile = Image.fromarray(rgb, mode="RGB").resize((900, 520), resample=Image.Resampling.BICUBIC)

    left_pad = 90
    right_pad = 20
    top_pad = 58
    bottom_pad = 72
    width = left_pad + tile.size[0] + right_pad
    height = top_pad + tile.size[1] + bottom_pad
    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    canvas.paste(tile, (left_pad, top_pad))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((12, 12), title, fill=(0, 0, 0), font=font)
    draw.text(
        (12, 28),
        f"symlog10(1 + |x| / {linthresh:.2e}), |transformed|max={vmax:.2f}",
        fill=(80, 80, 80),
        font=font,
    )
    draw.text((width // 2 - 48, height - 24), xlabel, fill=(0, 0, 0), font=font)
    draw.text((12, top_pad + tile.size[1] // 2), "time", fill=(0, 0, 0), font=font)

    for frac in [0.0, 0.5, 1.0]:
        x = left_pad + int(frac * (tile.size[0] - 1))
        idx = min(len(x_values) - 1, max(0, int(round(frac * (len(x_values) - 1)))))
        draw.text((x - 16, height - 44), f"{x_values[idx]:.1f}", fill=(0, 0, 0), font=font)
    for frac in [0.0, 0.5, 1.0]:
        y = top_pad + int(frac * (tile.size[1] - 1))
        idx = min(len(t_values) - 1, max(0, int(round(frac * (len(t_values) - 1)))))
        draw.text((12, y - 6), f"{t_values[idx]:.2f}", fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _pick_time_indices(t_values: np.ndarray, targets: list[float]) -> list[tuple[float, int]]:
    picks = []
    for target in targets:
        idx = int(np.argmin(np.abs(t_values - target)))
        picks.append((float(t_values[idx]), idx))
    unique = []
    seen = set()
    for t, idx in picks:
        if idx not in seen:
            unique.append((t, idx))
            seen.add(idx)
    return unique


def _palette(n: int) -> list[tuple[int, int, int]]:
    base = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
        (23, 190, 207),
        (140, 86, 75),
        (227, 119, 194),
    ]
    if n <= len(base):
        return base[:n]
    colors = []
    for i in range(n):
        colors.append(base[i % len(base)])
    return colors


def _draw_lineout_panel(data: np.ndarray, x_values: np.ndarray, t_values: np.ndarray,
                        targets: list[float], out_path: Path, title: str, xlabel: str):
    picks = _pick_time_indices(t_values, targets)
    selected = np.stack([data[idx] for _, idx in picks], axis=0)
    selected = np.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)
    linthresh = _choose_linthresh(selected)
    transformed = _symlog_transform(selected, linthresh)
    ymax = float(np.max(np.abs(transformed[np.isfinite(transformed)]))) if np.any(np.isfinite(transformed)) else 1.0
    ymax = max(ymax, 1.0)

    width = 1100
    height = 700
    left_pad = 95
    right_pad = 180
    top_pad = 60
    bottom_pad = 75
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((12, 12), title, fill=(0, 0, 0), font=font)
    draw.text(
        (12, 28),
        f"y = symlog10(1 + |x| / {linthresh:.2e}), transformed range +/-{ymax:.2f}",
        fill=(80, 80, 80),
        font=font,
    )

    x0 = left_pad
    y0 = top_pad
    x1 = left_pad + plot_w
    y1 = top_pad + plot_h
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=1)

    y_mid = y0 + plot_h // 2
    draw.line([(x0, y_mid), (x1, y_mid)], fill=(160, 160, 160), width=1)

    for frac, label in [(0.0, -ymax), (0.5, 0.0), (1.0, ymax)]:
        y = y1 - int(frac * plot_h)
        draw.text((12, y - 6), f"{label:.1f}", fill=(0, 0, 0), font=font)

    for frac in [0.0, 0.5, 1.0]:
        x = x0 + int(frac * plot_w)
        idx = min(len(x_values) - 1, max(0, int(round(frac * (len(x_values) - 1)))))
        draw.text((x - 12, y1 + 10), f"{x_values[idx]:.1f}", fill=(0, 0, 0), font=font)

    draw.text((width // 2 - 40, height - 24), xlabel, fill=(0, 0, 0), font=font)
    draw.text((12, y0 + plot_h // 2 - 20), "symlog", fill=(0, 0, 0), font=font)

    colors = _palette(len(picks))
    x_span = float(x_values[-1] - x_values[0]) if len(x_values) > 1 else 1.0
    for (t_sel, idx), color, curve in zip(picks, colors, transformed):
        pts = []
        for xv, yv in zip(x_values, curve):
            px = x0 + int(round((xv - x_values[0]) / x_span * plot_w))
            py = y0 + int(round((ymax - yv) / (2.0 * ymax) * plot_h))
            pts.append((px, py))
        draw.line(pts, fill=color, width=2)

    legend_x = x1 + 16
    legend_y = y0 + 4
    for (t_sel, _), color in zip(picks, colors):
        draw.rectangle([legend_x, legend_y + 3, legend_x + 14, legend_y + 11], fill=color)
        draw.text((legend_x + 22, legend_y), f"t={t_sel:.2f}", fill=(0, 0, 0), font=font)
        legend_y += 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    args = _parse_args()
    archive_path = resolve_existing_output_path(args.archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing archive: {archive_path}")

    output_dir = (
        Path(normalize_output_dir(args.output_dir))
        if args.output_dir
        else archive_path.parent / "ke_budget_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(archive_path)
    t_values = np.asarray(data["t"], dtype=np.float64)
    k_bins = np.asarray(data["k_bins"], dtype=np.float64)

    plot_specs = [
        ("ke_nonlinear_shell_tendency", "KE Nonlinear Shell Transfer", "k"),
        ("ke_nonlinear_flux", "KE Nonlinear Cumulative Flux", "k"),
        ("ke_beta_shell_tendency", "KE Beta Shell Tendency", "k"),
        ("ke_stretch_shell_tendency", "KE Stretching Shell Tendency", "k"),
        ("ke_diss_shell_tendency", "KE Dissipation Shell Tendency", "k"),
        ("ke_total_shell_tendency", "KE Total Shell Tendency", "k"),
        ("w_nonlinear_shell_tendency", "w Variance Nonlinear Transfer", "k"),
        ("w_nonlinear_flux", "w Variance Nonlinear Cumulative Flux", "k"),
        ("w_q_coupling_shell_tendency", "w Variance q-Coupling Tendency", "k"),
        ("w_buoyancy_shell_tendency", "w Variance Buoyancy Tendency", "k"),
        ("w_buoyancy_shell_tendency_dealiased", "w Variance Buoyancy Tendency Dealiased", "k"),
        ("w_diss_shell_tendency", "w Variance Dissipation Tendency", "k"),
        ("w_total_shell_tendency", "w Variance Total Tendency", "k"),
        ("th_nonlinear_shell_tendency", "theta Variance Nonlinear Transfer", "k"),
        ("th_nonlinear_flux", "theta Variance Nonlinear Cumulative Flux", "k"),
        ("th_mean_feedback_shell_tendency", "theta Variance Mean-Feedback Tendency", "k"),
        ("th_mean_feedback_shell_tendency_dealiased", "theta Variance Mean-Feedback Tendency Dealiased", "k"),
        ("th_conduction_shell_tendency", "theta Variance Conduction Tendency", "k"),
        ("th_conduction_shell_tendency_dealiased", "theta Variance Conduction Tendency Dealiased", "k"),
        ("th_diss_shell_tendency", "theta Variance Dissipation Tendency", "k"),
        ("th_total_shell_tendency", "theta Variance Total Tendency", "k"),
        ("heat_flux_shell_dealiased", "Dealiased Heat-Flux Shell Contribution", "k"),
    ]

    for key, title, xlabel in plot_specs:
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float64)
        _draw_heatmap(arr, k_bins, t_values, output_dir / f"{key}_heatmap.png", title, xlabel)
        _draw_lineout_panel(
            arr,
            k_bins,
            t_values,
            args.times,
            output_dir / f"{key}_lineouts.png",
            f"{title} Selected Times",
            xlabel,
        )

    print(f"saved plots to {output_dir}")


if __name__ == "__main__":
    main()
