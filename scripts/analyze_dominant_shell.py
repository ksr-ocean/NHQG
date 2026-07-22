#!/usr/bin/env python
"""Extract and plot time series for the dominant horizontal shell."""

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
        "--field",
        type=str,
        default="ke_stretch_shell_tendency",
        help="Shell field used to define the dominant shell.",
    )
    parser.add_argument(
        "--reference-time",
        type=float,
        default=None,
        help="If omitted, use the final saved time.",
    )
    parser.add_argument(
        "--report-times",
        type=float,
        nargs="+",
        default=[3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0],
    )
    return parser.parse_args()


def _choose_index(times: np.ndarray, target: float | None) -> int:
    if target is None:
        return len(times) - 1
    return int(np.argmin(np.abs(times - target)))


def _pick_rows(times: np.ndarray, targets: list[float]) -> list[tuple[float, int]]:
    rows = []
    seen = set()
    for target in targets:
        idx = int(np.argmin(np.abs(times - target)))
        if idx not in seen:
            rows.append((float(times[idx]), idx))
            seen.add(idx)
    return rows


def _symlog_transform(values: np.ndarray, linthresh: float) -> np.ndarray:
    return np.sign(values) * np.log10(1.0 + np.abs(values) / linthresh)


def _choose_linthresh(arrays: list[np.ndarray]) -> float:
    finite = np.concatenate([np.abs(a[np.isfinite(a)]) for a in arrays if np.any(np.isfinite(a))])
    if finite.size == 0:
        return 1.0
    p50 = float(np.percentile(finite, 50.0))
    p90 = float(np.percentile(finite, 90.0))
    val = max(1e-300, min(p50, p90 / 10.0 if p90 > 0 else p50))
    return val if np.isfinite(val) and val > 0 else 1.0


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
                title: str):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(0, 0, 0), width=1)
    font = ImageFont.load_default()
    draw.text((x0 + 6, y0 + 6), title, fill=(0, 0, 0), font=font)

    clean_series = [(name, np.nan_to_num(values, nan=0.0)) for name, values in series]
    linthresh = _choose_linthresh([values for _, values in clean_series])
    transformed = [(name, _symlog_transform(values, linthresh)) for name, values in clean_series]
    ymax = max(float(np.max(np.abs(values))) for _, values in transformed)
    ymax = max(ymax, 1.0)

    left = x0 + 52
    right = x1 - 150
    top = y0 + 28
    bottom = y1 - 30

    y_mid = (top + bottom) // 2
    draw.line([(left, y_mid), (right, y_mid)], fill=(170, 170, 170), width=1)
    draw.line([(left, top), (left, bottom)], fill=(0, 0, 0), width=1)
    draw.line([(left, bottom), (right, bottom)], fill=(0, 0, 0), width=1)

    t0 = float(times[0])
    t1 = float(times[-1]) if len(times) > 1 else t0 + 1.0
    span = max(t1 - t0, 1e-12)

    for frac, label in [(0.0, ymax), (0.5, 0.0), (1.0, -ymax)]:
        y = top + int(frac * (bottom - top))
        draw.text((x0 + 4, y - 6), f"{label:.1f}", fill=(0, 0, 0), font=font)

    for frac in [0.0, 0.5, 1.0]:
        x = left + int(frac * (right - left))
        t_label = t0 + frac * span
        draw.text((x - 10, bottom + 8), f"{t_label:.1f}", fill=(0, 0, 0), font=font)

    colors = _palette(len(series))
    for (name, values), color in zip(transformed, colors):
        pts = []
        for t, yv in zip(times, values):
            px = left + int(round((float(t) - t0) / span * (right - left)))
            py = top + int(round((ymax - float(yv)) / (2.0 * ymax) * (bottom - top)))
            pts.append((px, py))
        draw.line(pts, fill=color, width=2)

    legend_x = right + 12
    legend_y = top
    draw.text((legend_x, y0 + 6), f"linthresh={linthresh:.1e}", fill=(80, 80, 80), font=font)
    for (name, _), color in zip(series, colors):
        draw.rectangle([legend_x, legend_y + 3, legend_x + 12, legend_y + 11], fill=color)
        draw.text((legend_x + 18, legend_y), name, fill=(0, 0, 0), font=font)
        legend_y += 18


def _write_summary(path: Path, archive: Path, field: str, ref_time: float,
                   shell_idx: int, shell_k: float, report_rows: list[tuple[float, int]],
                   data: dict[str, np.ndarray]):
    lines = []
    lines.append("# Dominant Shell Summary")
    lines.append("")
    lines.append(f"- archive: `{archive}`")
    lines.append(f"- selector field: `{field}`")
    lines.append(f"- reference time: `{ref_time:.2f}`")
    lines.append(f"- dominant shell index: `{shell_idx}`")
    lines.append(f"- dominant shell center: `k = {shell_k:.6f}`")
    lines.append("")
    lines.append("## Selected Times")
    lines.append("")
    header = (
        "| time | Nu_dealiased | Nu_raw | max_speed | ke_shell | w_shell | th_shell | ke_stretch | ke_diss | "
        "w_buoyancy_raw | w_buoyancy_dealias | w_q_coupling | th_conduction_raw | th_conduction_dealias | "
        "th_mean_feedback_raw | th_mean_feedback_dealias |"
    )
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for t_sel, idx in report_rows:
        lines.append(
            "| "
            f"{t_sel:.2f} | "
            f"{data['Nusselt_dealiased'][idx]:.6e} | "
            f"{data['Nusselt'][idx]:.6e} | "
            f"{data['max_speed'][idx]:.6e} | "
            f"{data['ke_shell'][idx]:.6e} | "
            f"{data['w_shell'][idx]:.6e} | "
            f"{data['th_shell'][idx]:.6e} | "
            f"{data['ke_stretch'][idx]:.6e} | "
            f"{data['ke_diss'][idx]:.6e} | "
            f"{data['w_buoyancy'][idx]:.6e} | "
            f"{data['w_buoyancy_dealiased'][idx]:.6e} | "
            f"{data['w_q_coupling'][idx]:.6e} | "
            f"{data['th_conduction'][idx]:.6e} | "
            f"{data['th_conduction_dealiased'][idx]:.6e} | "
            f"{data['th_mean_feedback'][idx]:.6e} | "
            f"{data['th_mean_feedback_dealiased'][idx]:.6e} |"
        )
    lines.append("")
    lines.append("## Mean Exchange Diagnostics")
    lines.append("")
    header = (
        "| time | mean_energy | mean_flux_exchange | th_mean_feedback_sum_raw | th_mean_feedback_sum_dealias | "
        "exchange_residual_raw | exchange_residual_dealias | mean_grad_min | mean_grad_mid | mean_grad_max |"
    )
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for t_sel, idx in report_rows:
        lines.append(
            "| "
            f"{t_sel:.2f} | "
            f"{data['mean_energy'][idx]:.6e} | "
            f"{data['mean_flux_exchange'][idx]:.6e} | "
            f"{data['th_mean_feedback_sum_global'][idx]:.6e} | "
            f"{data['th_mean_feedback_sum_dealiased_global'][idx]:.6e} | "
            f"{data['exchange_residual'][idx]:.6e} | "
            f"{data['exchange_residual_dealiased'][idx]:.6e} | "
            f"{data['mean_grad_min'][idx]:.6e} | "
            f"{data['mean_grad_mid'][idx]:.6e} | "
            f"{data['mean_grad_max'][idx]:.6e} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The dominant shell time series are intended to expose the low-shell source chain "
        "`theta conduction -> buoyancy into w -> q-w coupling into KE` directly, while "
        "the mean-exchange table checks whether the fluctuation mean-feedback term and "
        "the mean-reservoir flux term are closing as an internal transfer."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = _parse_args()
    archive = resolve_existing_output_path(args.archive)
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")

    out_dir = (
        Path(normalize_output_dir(args.output_dir))
        if args.output_dir
        else archive.parent / "dominant_shell"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(archive)
    times = np.asarray(npz["t"], dtype=np.float64)
    k_bins = np.asarray(npz["k_bins"], dtype=np.float64)
    ref_idx = _choose_index(times, args.reference_time)
    selector = np.asarray(npz[args.field], dtype=np.float64)[ref_idx]
    shell_idx = int(np.argmax(np.abs(selector)))
    shell_k = float(k_bins[shell_idx])

    def _scalar(name: str, default: float = np.nan) -> np.ndarray:
        if name in npz:
            return np.asarray(npz[name], dtype=np.float64)
        return np.full_like(times, default, dtype=np.float64)

    data = {
        "t": times,
        "k_bins": k_bins,
        "shell_index": np.array(shell_idx, dtype=np.int64),
        "shell_k": np.array(shell_k, dtype=np.float64),
        "Nusselt": _scalar("Nusselt"),
        "Nusselt_dealiased": _scalar("Nusselt_dealiased"),
        "max_speed": _scalar("max_speed"),
        "heat_flux_mismatch": _scalar("heat_flux_mismatch"),
        "mean_energy": _scalar("mean_energy"),
        "mean_flux_exchange": _scalar("mean_flux_exchange_tendency"),
        "th_mean_feedback_sum_global": _scalar("th_mean_feedback_sum"),
        "th_mean_feedback_sum_dealiased_global": _scalar("th_mean_feedback_sum_dealiased"),
        "exchange_residual": _scalar("mean_theta_exchange_residual"),
        "exchange_residual_rel": _scalar("mean_theta_exchange_residual_rel"),
        "exchange_residual_dealiased": _scalar("mean_theta_exchange_residual_dealiased"),
        "exchange_residual_dealiased_rel": _scalar("mean_theta_exchange_residual_dealiased_rel"),
        "mean_grad_min": _scalar("mean_grad_min"),
        "mean_grad_mid": _scalar("mean_grad_mid"),
        "mean_grad_max": _scalar("mean_grad_max"),
        "th_bar_phys_max": _scalar("th_bar_phys_max"),
        "dth_bar_dz_max": _scalar("dth_bar_dz_max"),
        "ke_shell": np.asarray(npz["ke_horiz_spec"], dtype=np.float64)[:, shell_idx],
        "q_shell": np.asarray(npz["q_horiz_spec"], dtype=np.float64)[:, shell_idx],
        "w_shell": np.asarray(npz["w_horiz_spec"], dtype=np.float64)[:, shell_idx],
        "th_shell": np.asarray(npz["th_horiz_spec"], dtype=np.float64)[:, shell_idx],
        "ke_total": np.asarray(npz["ke_total_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "ke_stretch": np.asarray(npz["ke_stretch_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "ke_diss": np.asarray(npz["ke_diss_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "ke_nonlinear": np.asarray(npz["ke_nonlinear_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "w_total": np.asarray(npz["w_total_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "w_buoyancy": np.asarray(npz["w_buoyancy_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "w_buoyancy_dealiased": _scalar("w_buoyancy_shell_tendency_dealiased")[..., shell_idx] if "w_buoyancy_shell_tendency_dealiased" in npz else np.full_like(times, np.nan, dtype=np.float64),
        "w_q_coupling": np.asarray(npz["w_q_coupling_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "w_diss": np.asarray(npz["w_diss_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "w_nonlinear": np.asarray(npz["w_nonlinear_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "th_total": np.asarray(npz["th_total_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "th_conduction": np.asarray(npz["th_conduction_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "th_conduction_dealiased": _scalar("th_conduction_shell_tendency_dealiased")[..., shell_idx] if "th_conduction_shell_tendency_dealiased" in npz else np.full_like(times, np.nan, dtype=np.float64),
        "th_mean_feedback": np.asarray(npz["th_mean_feedback_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "th_mean_feedback_dealiased": _scalar("th_mean_feedback_shell_tendency_dealiased")[..., shell_idx] if "th_mean_feedback_shell_tendency_dealiased" in npz else np.full_like(times, np.nan, dtype=np.float64),
        "th_diss": np.asarray(npz["th_diss_shell_tendency"], dtype=np.float64)[:, shell_idx],
        "th_nonlinear": np.asarray(npz["th_nonlinear_shell_tendency"], dtype=np.float64)[:, shell_idx],
    }
    if np.all(np.isnan(data["th_mean_feedback_sum_dealiased_global"])) and "th_mean_feedback_shell_tendency_dealiased" in npz:
        data["th_mean_feedback_sum_dealiased_global"] = np.sum(
            np.asarray(npz["th_mean_feedback_shell_tendency_dealiased"], dtype=np.float64), axis=1
        )

    np.savez(out_dir / "dominant_shell_timeseries.npz", **data)

    report_rows = _pick_rows(times, args.report_times)
    _write_summary(
        out_dir / "dominant_shell_summary.md",
        archive,
        args.field,
        float(times[ref_idx]),
        shell_idx,
        shell_k,
        report_rows,
        data,
    )

    width = 1200
    height = 1860
    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), f"Dominant Shell Analysis: k={shell_k:.6f}", fill=(0, 0, 0), font=font)
    draw.text(
        (12, 28),
        f"selector={args.field} at t={times[ref_idx]:.2f}, shell index={shell_idx}",
        fill=(80, 80, 80),
        font=font,
    )

    _draw_panel(
        draw,
        (20, 60, 1180, 340),
        times,
        [
            ("KE total", data["ke_total"]),
            ("KE stretch", data["ke_stretch"]),
            ("KE diss", data["ke_diss"]),
            ("KE nonlinear", data["ke_nonlinear"]),
        ],
        "KE Budget At Dominant Shell",
    )
    _draw_panel(
        draw,
        (20, 360, 1180, 640),
        times,
        [
            ("w total", data["w_total"]),
            ("w buoyancy", data["w_buoyancy"]),
            ("w q-coupling", data["w_q_coupling"]),
            ("w diss", data["w_diss"]),
        ],
        "w Variance Budget At Dominant Shell",
    )
    _draw_panel(
        draw,
        (20, 660, 1180, 940),
        times,
        [
            ("theta total", data["th_total"]),
            ("theta cond raw", data["th_conduction"]),
            ("theta cond deal", data["th_conduction_dealiased"]),
            ("theta mf raw", data["th_mean_feedback"]),
            ("theta mf deal", data["th_mean_feedback_dealiased"]),
            ("theta diss", data["th_diss"]),
        ],
        "theta Variance Budget At Dominant Shell",
    )
    _draw_panel(
        draw,
        (20, 960, 1180, 1240),
        times,
        [
            ("KE shell", data["ke_shell"]),
            ("q shell", data["q_shell"]),
            ("w shell", data["w_shell"]),
            ("theta shell", data["th_shell"]),
        ],
        "Dominant-Shell Amplitudes",
    )
    _draw_panel(
        draw,
        (20, 1260, 1180, 1540),
        times,
        [
            ("w buoy raw", data["w_buoyancy"]),
            ("w buoy deal", data["w_buoyancy_dealiased"]),
            ("heat-flux mismatch", data["heat_flux_mismatch"]),
        ],
        "Thermal Shell Comparison",
    )
    _draw_panel(
        draw,
        (20, 1560, 1180, 1840),
        times,
        [
            ("mean flux exch", data["mean_flux_exchange"]),
            ("theta mf raw", data["th_mean_feedback_sum_global"]),
            ("theta mf deal", data["th_mean_feedback_sum_dealiased_global"]),
            ("exchange raw", data["exchange_residual"]),
            ("exchange deal", data["exchange_residual_dealiased"]),
            ("heat-flux mismatch", data["heat_flux_mismatch"]),
        ],
        "Mean-Thermal Exchange Diagnostics",
    )
    canvas.save(out_dir / "dominant_shell_timeseries.png")
    print(f"saved dominant-shell analysis to {out_dir}")


if __name__ == "__main__":
    main()
