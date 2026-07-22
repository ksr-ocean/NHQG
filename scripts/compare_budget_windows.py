"""Compare radial spectral-budget windows for stationarity / convergence.

Reads budget.npz files produced by scripts/spectral_budget_radial.py and
overlays, per window:
  - horizontal KE spectra E(k)             (convergence criterion 1)
  - vertical KE pseudo-spectra E_m         (convergence criterion 2)
  - scalar variance spectra Theta(k), Theta_m  (convergence criterion 3)
  - energy / enstrophy fluxes Pi_E(k), Pi_Z(k)
  - stitched E_bt / E_tot / Z_tot time series (development check)

Usage:
  python scripts/compare_budget_windows.py --label "t=40-80" analysis/spectral_budget/window_t40_t80 \
      --label "t=80-120" analysis/spectral_budget/window_t80_t120 ... --out FIG.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", dest="labels", default=[])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Spectral-budget window comparison")
    args = ap.parse_args()
    labels = args.labels if len(args.labels) == len(args.dirs) else [
        os.path.basename(d.rstrip("/")) for d in args.dirs]

    wins = []
    for d, lab in zip(args.dirs, labels):
        z = np.load(os.path.join(d, "budget.npz"))
        wins.append((lab, z))

    colors = ["k", "C0", "C3", "C2", "C4", "C5"]

    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))

    def sel_of(z):
        return np.where(z["in_cut"])[0][1:]

    # (a) horizontal KE spectra
    ax = axs[0, 0]
    for i, (lab, z) in enumerate(wins):
        s = sel_of(z)
        ax.loglog(z["k_bins"][s], z["E_spec"][s], color=colors[i], lw=1.5, label=lab)
    ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$E(k)$")
    ax.set_title("Horizontal KE spectra"); ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    # (b) scalar variance spectra
    ax = axs[0, 1]
    for i, (lab, z) in enumerate(wins):
        s = sel_of(z)
        ax.loglog(z["k_bins"][s], z["TH_spec"][s], color=colors[i], lw=1.5, label=lab)
    ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$\frac{1}{2}\langle|\theta|^2\rangle(k)$")
    ax.set_title("Scalar (temperature) variance spectra"); ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    # (c) vertical KE pseudo-spectra
    ax = axs[0, 2]
    for i, (lab, z) in enumerate(wins):
        m = np.arange(len(z["vert_E"]))
        ax.loglog(m[1:], z["vert_E"][1:], color=colors[i], lw=1.5, label=lab)
    ax.set_xlabel("Chebyshev degree $m$"); ax.set_ylabel(r"$E_m$")
    ax.set_title("Vertical KE pseudo-spectra"); ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    # (d) vertical theta spectra (+ w faint)
    ax = axs[1, 0]
    for i, (lab, z) in enumerate(wins):
        m = np.arange(len(z["vert_TH"]))
        ax.loglog(m[1:], z["vert_TH"][1:], color=colors[i], lw=1.5, label=lab)
        ax.loglog(m[1:], z["vert_W"][1:], color=colors[i], lw=0.7, ls=":", alpha=0.7)
    ax.set_xlabel("Chebyshev degree $m$"); ax.set_ylabel(r"$\Theta_m$ (solid), $W_m$ (dotted)")
    ax.set_title(r"Vertical $\theta$ (and $w$) pseudo-spectra"); ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    # (e) energy flux
    ax = axs[1, 1]
    for i, (lab, z) in enumerate(wins):
        s = sel_of(z)
        ax.semilogx(z["k_bins"][s], -np.cumsum(z["E_adv"][s]), color=colors[i], lw=1.5, label=lab)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$\Pi_E(k)$")
    ax.set_title(r"Energy flux ($<0$: inverse cascade)"); ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    # (f) enstrophy flux, or time series if series available
    ax = axs[1, 2]
    for i, (lab, z) in enumerate(wins):
        s = sel_of(z)
        ax.semilogx(z["k_bins"][s], -np.cumsum(z["Z_adv"][s]), color=colors[i], lw=1.5, label=lab)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$\Pi_Z(k)$")
    ax.set_title(r"Enstrophy flux ($>0$: forward cascade)"); ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    fig.suptitle(args.title, y=0.995)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"wrote {args.out}")

    # quantitative convergence metrics between consecutive windows
    print("\nconvergence metrics (max |log10 ratio| over plotted shells/modes):")
    for a in range(len(wins) - 1):
        (la, za), (lb, zb) = wins[a], wins[a + 1]
        s = sel_of(za)
        for key, lab in (("E_spec", "E(k)"), ("TH_spec", "Th(k)"),
                         ("vert_E", "E_m"), ("vert_TH", "Th_m")):
            xa, xb = za[key], zb[key]
            if key.startswith("vert"):
                xa, xb = xa[1:], xb[1:]
            else:
                xa, xb = xa[s], xb[s]
            good = (xa > 0) & (xb > 0)
            r = np.max(np.abs(np.log10(xa[good] / xb[good])))
            print(f"  {la} vs {lb}: {lab:6s} max|log10 ratio| = {r:.3f} "
                  f"(x{10**r:.2f})")


if __name__ == "__main__":
    main()
