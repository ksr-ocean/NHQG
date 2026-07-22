"""Radial (angle-averaged) spectral budgets for the vorticity equation.

For the q equation as integrated by the production solver (beta=0, Ld=inf,
so q = zeta = del_perp^2 psi):

    dq/dt = -div(u q)  +  dw/dZ  -  nu |k|^2 q
            (advection)  (stretch)  (dissipation)

Budgets, per horizontal-wavenumber shell |k| in [i dk, (i+1) dk), dk = 2pi/L,
depth-integrated with Clenshaw-Curtis weights:

    Energy    E(k) = 1/2 |k|^2 |psi_hat|^2 :  dE/dt term = -Re( psi_hat^* T_hat )
              (q_hat = -|k|^2 psi_hat, so multiplying the q equation by
               psi_hat^* and flipping sign gives the KE budget)
    Enstrophy Z(k) = 1/2 |q_hat|^2         :  dZ/dt term = +Re( q_hat^* T_hat )

Fluxes: Pi(k) = -sum_{k' <= k} T_adv(k').  Pi > 0: forward cascade through k;
Pi < 0: inverse cascade.

Ghost hygiene (see hermitian_ghost.md): the anti-Hermitian ky=0 ghost — the
secular, linearly growing, physically invisible mode — is projected out of
every field before any computation, and the removed magnitude is reported.
State content outside the 2/3 dealiasing mask is measured and zeroed so the
budget matches the band-limited system the solver actually integrates.

Barotropic split: psi_bt = depth average of psi. The stretching term
depth-integrates to w(1)-w(0) = 0 exactly, so the barotropic KE budget
isolates the nonlinear baroclinic->barotropic transfer that feeds the LSV.

Products use the run's 2/3-rule path (unpadded FFT + output mask), matching
nhqg/spectral.py:_triple_flux_divergence_one_level_23 bit-for-bit in exact
arithmetic; startup self-tests verify operators and exact conservation.

Usage:
  python scripts/spectral_budget_radial.py \
      --run-dir output/output_combined_Nx128_Nz256_t40_to_t70_sub4_23rule_stack13_snap01 \
      --t-min 100 --t-max 120 --stride 2 \
      --out analysis/spectral_budget/window_t100_t120
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

L_C = 2.0 * np.pi / 1.3048  # critical wavelength; domain L = 10 L_c


# ---------------------------------------------------------------------------
# Vertical operators (numpy mirrors of nhqg/grid.py; self-tested below)
# ---------------------------------------------------------------------------

def cheb_vandermonde(N):
    """V[j, n] = T_n(xi_j), xi_j = cos(pi j / N)."""
    j = np.arange(N + 1)
    n = np.arange(N + 1)
    return np.cos(np.pi * np.outer(j, n) / N)


def cheb_vandermonde_inv(N):
    """Analytic DCT-I inverse: V_inv[n, j] = 2 / (N c_n c_j) cos(n pi j / N)."""
    c = np.ones(N + 1)
    c[0] = c[N] = 2.0
    n = np.arange(N + 1)
    j = np.arange(N + 1)
    return (2.0 / N) * np.cos(np.pi * np.outer(n, j) / N) / np.outer(c, c)


def cheb_diff_matrix_coeff(N):
    """G_xi: Chebyshev-coefficient first-derivative matrix on xi in [-1, 1]."""
    G = np.zeros((N + 1, N + 1))
    # b_{n} = (2/c_n) * sum_{p = n+1, n+3, ...} p a_p
    for n in range(N + 1):
        cn = 2.0 if n == 0 else 1.0
        for p in range(n + 1, N + 1, 2):
            G[n, p] = 2.0 * p / cn
    return G


def cc_weights_unit(N):
    """Clenshaw-Curtis weights for CGL nodes mapped to Z in [0, 1]."""
    # Standard CC on xi in [-1, 1] (with b_k boundary correction), halved.
    w = np.zeros(N + 1)
    jj = np.arange(N + 1)
    v = np.zeros(N + 1)
    for j in range(N + 1):
        s = 0.0
        for k in range(1, N // 2 + 1):
            b = 1.0 if 2 * k == N else 2.0
            s += b / (4.0 * k * k - 1.0) * np.cos(2.0 * k * np.pi * j / N)
        v[j] = 1.0 - s
    c = np.ones(N + 1)
    c[0] = c[N] = 0.5
    w = 2.0 * c * v / N
    return 0.5 * w  # map [-1,1] -> [0,1]


def dirichlet_stencil(N):
    """S[(N+1), (N-1)]: Galerkin basis phi_j = -T_j + T_{j+2}."""
    S = np.zeros((N + 1, N - 1))
    for j in range(N - 1):
        S[j, j] = -1.0
        S[j + 2, j] = 1.0
    return S


# ---------------------------------------------------------------------------
# Horizontal machinery
# ---------------------------------------------------------------------------

def wavenumbers(Nx, L):
    k0 = 2.0 * np.pi / L
    kx_int = np.fft.fftfreq(Nx, d=1.0 / Nx)          # 0..Nx/2, -Nx/2+1..-1 (fftfreq: 0..63,-64..-1 for 128? -> 0..63, -64..-1; Nyquist at -64)
    ky_int = np.arange(Nx // 2 + 1)
    kx = k0 * kx_int[:, None]
    ky = k0 * ky_int[None, :]
    ksq = kx ** 2 + ky ** 2
    return k0, kx_int, ky_int, kx, ky, ksq


def mask_23(Nx, kx_int, ky_int):
    K = Nx // 3
    return ((np.abs(kx_int)[:, None] <= K) & (ky_int[None, :] <= K)).astype(float)


def rfft_weights(Nx):
    """Half-plane multiplicity: 2 for 0 < ky < Nyquist, 1 at ky = 0 and Nyquist."""
    Nk = Nx // 2 + 1
    w = 2.0 * np.ones(Nk)
    w[0] = 1.0
    w[-1] = 1.0
    return w


def hermitian_project_ky0(f, Nx):
    """Project the ky=0 and ky=Nyquist columns onto the Hermitian subspace.

    Returns (projected field, max |anti-Hermitian part| removed)."""
    neg = (-np.arange(Nx)) % Nx
    ghost = 0.0
    out = f.copy()
    for col in (0, f.shape[-1] - 1):
        a = f[..., :, col]
        h = 0.5 * (a + np.conj(a[..., neg]))
        ghost = max(ghost, float(np.max(np.abs(a - h))))
        out[..., :, col] = h
    return out, ghost


def advective_tendency_32(psi_n, q_n, kx, ky, Nx):
    """T_adv = -div(u q), 3/2-rule (padded products), batched in Z.

    Mirrors nhqg/spectral.py conservative_flux_divergence_dealiased: pad each
    factor, multiply on the padded grid, truncate with the (Npad/Nx)^2
    correction, apply the divergence on the Nx grid."""
    Npad = (3 * Nx) // 2
    Nk = Nx // 2 + 1
    Nk_pad = Npad // 2 + 1

    def pad(f):
        out = np.zeros(f.shape[:-2] + (Npad, Nk_pad), dtype=f.dtype)
        out[..., :Nx // 2, :Nk] = f[..., :Nx // 2, :]
        out[..., Npad - Nx // 2:, :Nk] = f[..., Nx // 2:, :]
        return out

    def trunc(f):
        out = np.empty(f.shape[:-2] + (Nx, Nk), dtype=f.dtype)
        out[..., :Nx // 2, :] = f[..., :Nx // 2, :Nk]
        out[..., Nx // 2:, :] = f[..., Npad - Nx // 2:, :Nk]
        return out * (float(Npad) / Nx) ** 2

    u = np.fft.irfft2(pad(-1j * ky * psi_n), s=(Npad, Npad), axes=(-2, -1))
    v = np.fft.irfft2(pad(1j * kx * psi_n), s=(Npad, Npad), axes=(-2, -1))
    qp = np.fft.irfft2(pad(q_n), s=(Npad, Npad), axes=(-2, -1))
    uf = trunc(np.fft.rfft2(u * qp, axes=(-2, -1)))
    vf = trunc(np.fft.rfft2(v * qp, axes=(-2, -1)))
    return -(1j * kx * uf + 1j * ky * vf)


def advective_tendency_23(psi_n, q_n, kx, ky, mask, Nx):
    """T_adv = -div(u q), 2/3-rule (unpadded FFT + output mask), batched in Z.

    Mirrors nhqg/spectral.py:_triple_flux_divergence_one_level_23 with the
    solver's sign (explicit tendency E_q = -A_q)."""
    u = np.fft.irfft2(-1j * ky * psi_n, s=(Nx, Nx), axes=(-2, -1))
    v = np.fft.irfft2(1j * kx * psi_n, s=(Nx, Nx), axes=(-2, -1))
    qp = np.fft.irfft2(q_n, s=(Nx, Nx), axes=(-2, -1))
    uf = np.fft.rfft2(u * qp, axes=(-2, -1))
    vf = np.fft.rfft2(v * qp, axes=(-2, -1))
    return -(1j * kx * uf + 1j * ky * vf) * mask


# ---------------------------------------------------------------------------
# Shell reduction
# ---------------------------------------------------------------------------

def shell_reduce(field2d, bin_idx, n_bins):
    """Sum a real (Nx, Nk) array onto radial shells."""
    return np.bincount(bin_idx.ravel(), weights=field2d.ravel(), minlength=n_bins)[:n_bins]


def vmat(M, X):
    """BLAS-backed contraction M[a,b] X[b,x,y] -> [a,x,y] (einsum is too slow here)."""
    b, nx, nk = X.shape
    return (M @ X.reshape(b, nx * nk)).reshape(M.shape[0], nx, nk)


def depth_dot(w, X):
    """BLAS-backed contraction w[j] X[j,x,y] -> [x,y]."""
    j, nx, nk = X.shape
    return (w @ X.reshape(j, nx * nk)).reshape(nx, nk)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def run_self_tests(Nz, Nx, L):
    rng = np.random.default_rng(0)
    V = cheb_vandermonde(Nz)
    Vi = cheb_vandermonde_inv(Nz)
    GZ = 2.0 * cheb_diff_matrix_coeff(Nz)
    ccw = cc_weights_unit(Nz)
    S = dirichlet_stencil(Nz)
    xi = np.cos(np.pi * np.arange(Nz + 1) / Nz)
    Zg = 0.5 * (1.0 + xi)

    # V / V_inv roundtrip
    assert np.max(np.abs(V @ Vi - np.eye(Nz + 1))) < 1e-10, "V/V_inv roundtrip"

    # d/dZ sin(pi Z) = pi cos(pi Z)
    f = np.sin(np.pi * Zg)
    df = V @ (GZ @ (Vi @ f))
    assert np.max(np.abs(df - np.pi * np.cos(np.pi * Zg))) < 1e-7, "G_Z accuracy"

    # CC quadrature: int_0^1 sin(pi Z) dZ = 2/pi
    assert abs(ccw @ f - 2.0 / np.pi) < 1e-12, "CC weights"
    assert abs(np.sum(ccw) - 1.0) < 1e-12, "CC normalization"

    # Dirichlet stencil: lifted field vanishes at the walls
    wg = rng.standard_normal(Nz - 1)
    wn = V @ (S @ wg)
    assert max(abs(wn[0]), abs(wn[-1])) < 1e-10, "Dirichlet stencil BCs"

    # Exact conservation of the 2/3-rule advective term for band-limited,
    # Hermitian fields: sum_k Re(q^* T_adv) = 0 and sum_k -Re(psi^* T_adv) = 0
    k0, kx_int, ky_int, kx, ky, ksq = wavenumbers(Nx, L)
    m = mask_23(Nx, kx_int, ky_int)
    wr = rfft_weights(Nx)
    qp_phys = rng.standard_normal((2, Nx, Nx))
    q_n = np.fft.rfft2(qp_phys, axes=(-2, -1)) * m
    q_n[..., 0, 0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        psi_n = np.where(ksq > 0, -q_n / ksq, 0.0)
    T = advective_tendency_23(psi_n, q_n, kx, ky, m, Nx)
    zsum = np.sum(np.real(np.conj(q_n) * T) * wr[None, None, :])
    esum = np.sum(-np.real(np.conj(psi_n) * T) * wr[None, None, :])
    zmag = np.sum(np.abs(np.real(np.conj(q_n) * T)) * wr[None, None, :])
    emag = np.sum(np.abs(np.real(np.conj(psi_n) * T)) * wr[None, None, :])
    assert abs(zsum) < 1e-9 * zmag, f"enstrophy conservation: {zsum/zmag:.2e}"
    assert abs(esum) < 1e-9 * emag, f"energy conservation: {esum/emag:.2e}"

    # 3/2 path: conservation for Nyquist-free Hermitian fields. (With
    # Npad = 3Nx/2 exactly, Nyquist-shell self-interactions still alias —
    # a property of the production 3/2 path itself, see the 2026-07-03
    # review; physical states carry negligible Nyquist content.)
    q_f = np.fft.rfft2(rng.standard_normal((2, Nx, Nx)), axes=(-2, -1))
    q_f[..., 0, 0] = 0.0
    q_f[..., Nx // 2, :] = 0.0     # kx Nyquist row
    q_f[..., :, -1] = 0.0          # ky Nyquist column
    with np.errstate(divide="ignore", invalid="ignore"):
        psi_f = np.where(ksq > 0, -q_f / ksq, 0.0)
    T32 = advective_tendency_32(psi_f, q_f, kx, ky, Nx)
    z32 = np.sum(np.real(np.conj(q_f) * T32) * wr[None, None, :])
    z32m = np.sum(np.abs(np.real(np.conj(q_f) * T32)) * wr[None, None, :])
    e32 = np.sum(-np.real(np.conj(psi_f) * T32) * wr[None, None, :])
    e32m = np.sum(np.abs(np.real(np.conj(psi_f) * T32)) * wr[None, None, :])
    assert abs(z32) < 1e-9 * z32m, f"32-rule enstrophy conservation: {z32/z32m:.2e}"
    assert abs(e32) < 1e-9 * e32m, f"32-rule energy conservation: {e32/e32m:.2e}"
    print("[self-test] all operator and conservation checks passed (23 + 32 paths)")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--t-min", type=float, default=100.0)
    ap.add_argument("--t-max", type=float, default=120.0)
    ap.add_argument("--stride", type=int, default=2,
                    help="use every stride-th checkpoint in the window")
    ap.add_argument("--dt", type=float, default=5e-5)
    ap.add_argument("--nu", type=float, default=1.0)
    ap.add_argument("--L", type=float, default=10.0 * L_C)
    ap.add_argument("--dealias", choices=["23", "32"], default="23",
                    help="product path of the analyzed run: 23_rule (default) or 32_rule")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- inventory checkpoints in window ---------------------------------
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "checkpoint_*.npz")))
    picked = []
    for p in ckpts:
        step = int(re.search(r"checkpoint_(\d+)\.npz", p).group(1))
        t = step * args.dt
        if args.t_min - 1e-9 <= t <= args.t_max + 1e-9:
            picked.append((t, p))
    picked = picked[:: args.stride]
    if not picked:
        sys.exit("no checkpoints in window")
    print(f"[info] {len(picked)} checkpoints, t = {picked[0][0]:.2f} .. {picked[-1][0]:.2f}")

    # ---- probe shapes ------------------------------------------------------
    d0 = np.load(picked[0][1])
    Nz1, Nx, Nk = d0["q_hat_real"].shape
    Nz = Nz1 - 1
    print(f"[info] Nx={Nx}, Nz={Nz}, L={args.L:.4f}")

    run_self_tests(Nz, Nx, args.L)

    # ---- operators ---------------------------------------------------------
    V = cheb_vandermonde(Nz)
    Vi = cheb_vandermonde_inv(Nz)
    GZ = 2.0 * cheb_diff_matrix_coeff(Nz)
    ccw = cc_weights_unit(Nz)
    S = dirichlet_stencil(Nz)
    GZS = GZ @ S  # Galerkin w -> Chebyshev coeffs of dw/dZ

    k0, kx_int, ky_int, kx, ky, ksq = wavenumbers(Nx, args.L)
    m23 = mask_23(Nx, kx_int, ky_int)
    wr = rfft_weights(Nx)
    kmag = np.sqrt(ksq)
    bin_idx = np.floor(kmag / k0 + 1e-9).astype(int)
    n_bins = int(bin_idx.max()) + 1
    k_bins = (np.arange(n_bins) + 0.5) * k0
    k_cut = (Nx // 3) * k0
    norm = float(Nx) ** 4

    def reduce_shells(prod_nodal):
        """Depth-integrate a real (Nz+1, Nx, Nk) product, weight, shell-bin."""
        depth = depth_dot(ccw, prod_nodal)
        return shell_reduce(depth * wr[None, :] / norm, bin_idx, n_bins)

    # ---- accumulators ------------------------------------------------------
    NAMES = ["E_spec", "E_bt_spec", "Z_spec", "TH_spec",
             "E_adv", "E_str", "E_diss",
             "Z_adv", "Z_str", "Z_diss",
             "Ebt_adv", "Ebt_diss"]
    acc = {name: np.zeros(n_bins) for name in NAMES}
    acc2 = {name: np.zeros(n_bins) for name in NAMES}
    series = {name: [] for name in ["t", "E_tot", "E_bt", "Z_tot",
                                    "E_str_sum", "E_adv_sum", "Ebt_adv_sum"]}
    spec_series = {"E": [], "Z": [], "Ebt": []}
    vert = {"E": np.zeros(Nz + 1), "TH": np.zeros(Nz + 1), "W": np.zeros(Nz + 1)}
    tails = {"q": 0.0, "psi": 0.0, "w": 0.0}

    def put(name, val):
        acc[name] += val
        acc2[name] += val * val
    ghost_max = 0.0
    outmask_frac_max = 0.0
    strbt_max = 0.0  # depth-integrated stretching (should be ~0)
    E_first = E_last = None

    for it, (t, path) in enumerate(picked):
        d = np.load(path)
        q_c = d["q_hat_real"] + 1j * d["q_hat_imag"]      # (Nz+1, Nx, Nk) Cheb coeffs
        w_g = d["w_hat_real"] + 1j * d["w_hat_imag"]      # (Nz-1, ...) Galerkin
        th_g = d["th_hat_real"] + 1j * d["th_hat_imag"]   # (Nz-1, ...) Galerkin

        # -- ghost projection (the secular linearly growing mode) ----------
        q_c, g1 = hermitian_project_ky0(q_c, Nx)
        w_g, g2 = hermitian_project_ky0(w_g, Nx)
        th_g, g3 = hermitian_project_ky0(th_g, Nx)
        ghost_max = max(ghost_max, g1, g2, g3)

        if args.dealias == "23":
            # -- band-limit to the run's 2/3 mask; measure what was outside --
            e_all = np.sum(np.abs(q_c) ** 2)
            e_out = np.sum((np.abs(q_c) ** 2) * (1.0 - m23)[None, :, :])
            outmask_frac_max = max(outmask_frac_max, float(e_out / max(e_all, 1e-300)))
            q_c = q_c * m23[None, :, :]
            w_g = w_g * m23[None, :, :]
            th_g = th_g * m23[None, :, :]

        # -- psi inversion (Ld = inf): psi = -q / |k|^2 ---------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            psi_c = np.where(ksq[None, :, :] > 0, -q_c / ksq[None, :, :], 0.0)

        # -- nodal fields ----------------------------------------------------
        q_n = vmat(V, q_c)
        psi_n = vmat(V, psi_c)
        dwdZ_n = vmat(V, vmat(GZS, w_g))

        # -- tendencies ------------------------------------------------------
        if args.dealias == "23":
            T_adv = advective_tendency_23(psi_n, q_n, kx, ky, m23, Nx)
        else:
            T_adv = advective_tendency_32(psi_n, q_n, kx, ky, Nx)
        T_str = dwdZ_n
        T_diss = -args.nu * ksq[None, :, :] * q_n

        # -- spectra ---------------------------------------------------------
        E_k = reduce_shells(0.5 * ksq[None, :, :] * np.abs(psi_n) ** 2)
        put("E_spec", E_k)
        Z_k = reduce_shells(0.5 * np.abs(q_n) ** 2)
        put("Z_spec", Z_k)
        psi_bt = depth_dot(ccw, psi_n)
        Ebt_k = shell_reduce(
            0.5 * ksq * np.abs(psi_bt) ** 2 * wr[None, :] / norm, bin_idx, n_bins)
        put("E_bt_spec", Ebt_k)
        spec_series["E"].append(E_k.astype(np.float32))
        spec_series["Z"].append(Z_k.astype(np.float32))
        spec_series["Ebt"].append(Ebt_k.astype(np.float32))

        # scalar (theta) variance spectrum, horizontal shells
        th_c = vmat(S, th_g)
        th_n = vmat(V, th_c)
        put("TH_spec", reduce_shells(0.5 * np.abs(th_n) ** 2))

        # vertical Chebyshev pseudo-spectra (per coefficient degree m)
        w_c_full = vmat(S, w_g)
        hw = wr[None, None, :] / norm
        vert["E"] += np.sum(0.5 * ksq[None, :, :] * np.abs(psi_c) ** 2 * hw, axis=(1, 2))
        vert["TH"] += np.sum(0.5 * np.abs(th_c) ** 2 * hw, axis=(1, 2))
        vert["W"] += np.sum(0.5 * np.abs(w_c_full) ** 2 * hw, axis=(1, 2))

        # vertical Chebyshev tail fractions (top 16 modes) — resolution audit
        for nm, coeffs in (("q", q_c), ("psi", psi_c),
                           ("w", np.einsum("nm,mxy->nxy", S, w_g) if False else None)):
            if coeffs is None:
                continue
            en = np.sum(np.abs(coeffs) ** 2, axis=(1, 2))
            tails[nm] = max(tails[nm], float(np.sum(en[-16:]) / max(np.sum(en), 1e-300)))
        if it == 0:
            E_first = E_k.copy()
        E_last = E_k.copy()

        # -- budgets ---------------------------------------------------------
        shot = {}
        for name, T in (("adv", T_adv), ("str", T_str), ("diss", T_diss)):
            eterm = reduce_shells(-np.real(np.conj(psi_n) * T))
            zterm = reduce_shells(np.real(np.conj(q_n) * T))
            put(f"E_{name}", eterm)
            put(f"Z_{name}", zterm)
            shot[name] = eterm

        # -- barotropic budget ----------------------------------------------
        T_adv_bt = depth_dot(ccw, T_adv)
        T_diss_bt = depth_dot(ccw, T_diss)
        T_str_bt = depth_dot(ccw, T_str)
        strbt_max = max(strbt_max, float(np.max(np.abs(T_str_bt))))
        ebt_adv = shell_reduce(
            -np.real(np.conj(psi_bt) * T_adv_bt) * wr[None, :] / norm, bin_idx, n_bins)
        ebt_diss = shell_reduce(
            -np.real(np.conj(psi_bt) * T_diss_bt) * wr[None, :] / norm, bin_idx, n_bins)
        put("Ebt_adv", ebt_adv)
        put("Ebt_diss", ebt_diss)

        series["t"].append(t)
        series["E_tot"].append(float(np.sum(E_k)))
        series["E_bt"].append(float(np.sum(Ebt_k)))
        series["Z_tot"].append(float(np.sum(Z_k)))
        series["E_str_sum"].append(float(np.sum(shot["str"])))
        series["E_adv_sum"].append(float(np.sum(shot["adv"])))
        series["Ebt_adv_sum"].append(float(np.sum(ebt_adv)))

        if (it + 1) % 10 == 0 or it == len(picked) - 1:
            print(f"[info] processed {it + 1}/{len(picked)} (t={t:.2f})")

    nsnap = len(picked)
    se = {}
    for k in acc:
        acc[k] /= nsnap
        var = np.maximum(acc2[k] / nsnap - acc[k] ** 2, 0.0)
        se[k] = np.sqrt(var / nsnap)

    dEdt_fd = (E_last - E_first) / (picked[-1][0] - picked[0][0])

    # ---- closure / hygiene report -----------------------------------------
    in_cut = k_bins <= k_cut + 0.5 * k0
    rep = []
    rep.append(f"window: t = {picked[0][0]:.2f} .. {picked[-1][0]:.2f}, "
               f"{len(picked)} snapshots, Nx={Nx}, Nz={Nz}")
    rep.append(f"ghost removed (max |anti-Hermitian ky=0| over window): {ghost_max:.3e}")
    rep.append(f"state energy outside 2/3 mask (max fraction): {outmask_frac_max:.3e}")
    rep.append(f"depth-integrated stretching in bt budget (max, should be ~0): {strbtmax_fmt(strbt_max)}")
    rep.append(f"vertical Chebyshev tail (top 16 modes / total, max over window): "
               f"q {tails['q']:.3e}, psi {tails['psi']:.3e}")
    for nm, lab in (("E", "energy"), ("Z", "enstrophy")):
        a = np.sum(acc[f"{nm}_adv"][in_cut])
        s = np.sum(acc[f"{nm}_str"][in_cut])
        dd = np.sum(acc[f"{nm}_diss"][in_cut])
        a_se = np.sqrt(np.sum(se[f"{nm}_adv"][in_cut] ** 2))
        s_se = np.sqrt(np.sum(se[f"{nm}_str"][in_cut] ** 2))
        amag = np.sum(np.abs(acc[f"{nm}_adv"][in_cut]))
        rep.append(f"{lab}: sum adv = {a:+.4e} +/- {a_se:.2e} (|.|-sum {amag:.3e}), "
                   f"sum stretch = {s:+.4e} +/- {s_se:.2e}, sum diss = {dd:+.4e}, "
                   f"net = {a+s+dd:+.4e}")
    ba = np.sum(acc["Ebt_adv"][in_cut]); ba_se = np.sqrt(np.sum(se["Ebt_adv"][in_cut] ** 2))
    bd = np.sum(acc["Ebt_diss"][in_cut])
    rep.append(f"barotropic: sum adv = {ba:+.4e} +/- {ba_se:.2e}, sum diss = {bd:+.4e}, "
               f"net = {ba+bd:+.4e}")
    rep.append(f"energy: <dE/dt> from endpoints = {np.sum(dEdt_fd[in_cut]):+.4e} "
               f"(vs budget net above; both should be small vs injection)")
    report = "\n".join(rep)
    print(report)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(report + "\n")

    np.savez(os.path.join(args.out, "budget.npz"),
             k_bins=k_bins, k_cut=k_cut, k0=k0, in_cut=in_cut,
             dEdt_fd=dEdt_fd, n_snapshots=len(picked),
             t_min=picked[0][0], t_max=picked[-1][0],
             **acc,
             **{f"se_{k}": v for k, v in se.items()},
             **{f"ts_{k}": np.array(v) for k, v in series.items()},
             **{f"spec_series_{k}": np.array(v) for k, v in spec_series.items()},
             **{f"vert_{k}": v / nsnap for k, v in vert.items()})

    make_plots(args.out, k_bins, k_cut, acc, in_cut, se, series, spec_series)
    print(f"[info] wrote {args.out}/budget.npz, summary.txt, and figures")


def strbtmax_fmt(x):
    return f"{x:.3e}"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(outdir, k_bins, k_cut, acc, in_cut, se, series, spec_series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = np.where(in_cut)[0][1:]   # drop the k=0 shell (empty bin wrecks log axes)
    k = k_bins[sel]
    kc_onset = 1.3048

    def cut(x):
        return x[sel]

    def style(ax, xlab=True):
        ax.axvline(kc_onset, color="0.6", lw=0.8, ls=":")
        ax.axvline(k_cut, color="0.6", lw=0.8, ls="--")
        ax.grid(alpha=0.25, which="both")
        if xlab:
            ax.set_xlabel(r"$k$")

    def flux(T):
        return -np.cumsum(cut(T))

    # ============================ ENERGY ================================
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axs[0]
    E = cut(acc["E_spec"])
    Ebt = cut(acc["E_bt_spec"])
    Ebc = np.maximum(E - Ebt, 1e-300)
    ax.loglog(k, E, "k-", lw=1.8, label=r"$E(k)$ total")
    ax.loglog(k, Ebt, "C0-", lw=1.4, label=r"$E_{bt}(k)$ barotropic")
    ax.loglog(k, Ebc, "C3-", lw=1.4, label=r"$E_{bc}(k)$ baroclinic")
    pos = E > 0
    kref = k[pos]
    if len(kref) > 4:
        i0 = max(1, np.argmax(E))
        for slope, lab, c in ((-5.0 / 3.0, r"$k^{-5/3}$", "0.4"), (-3.0, r"$k^{-3}$", "0.7")):
            ref = E[i0] * (kref / kref[i0]) ** slope
            ax.loglog(kref, ref, color=c, lw=0.8, ls="--")
            ax.annotate(lab, (kref[-1], ref[-1]), fontsize=8, color=c)
    style(ax)
    ax.set_ylabel(r"$E(k)$")
    ax.set_title("KE spectra (time-averaged)")
    ax.legend(fontsize=8)

    ax = axs[1]
    for nm, lab, c, lw, al in (
            ("E_adv", "advection (transfer)", "C0", 1.5, 1.0),
            ("E_diss", r"dissipation $-\nu k^2$", "C2", 1.5, 1.0),
            ("E_str", r"stretching, raw sample mean (aliased at low $k$)", "C3", 0.9, 0.55)):
        ax.semilogx(k, k * cut(acc[nm]), color=c, lw=lw, alpha=al, label=lab)
        ax.fill_between(k, k * cut(acc[nm] - se[nm]), k * cut(acc[nm] + se[nm]),
                        color=c, alpha=0.12, lw=0)
    # closure-inferred mean conversion: per-shell linear-fit dE/dt minus adv minus diss.
    # The direct sample mean of the stretch term is aliased by low-k oscillatory
    # modes at the 0.25 checkpoint cadence; this estimator uses the (slow) trend
    # of E(k, t) instead and has far smaller variance.
    ts_t = np.asarray(series["t"])
    Eser = np.asarray(spec_series["E"], dtype=float)
    slope = np.polyfit(ts_t, Eser, 1)[0]
    inferred = slope - acc["E_adv"] - acc["E_diss"]
    ax.semilogx(k, k * cut(inferred), color="C1", lw=1.9, ls="-",
                label="stretching (closure-inferred)")
    ax.semilogx(k, k * cut(slope), "k--", lw=1.0, label=r"$dE/dt$ (fit)")
    ylim = 1.4 * max(np.max(np.abs(k * cut(acc["E_adv"]))),
                     np.max(np.abs(k * cut(acc["E_diss"]))),
                     np.max(np.abs(k * cut(inferred))))
    ax.set_ylim(-ylim, ylim)
    ax.axhline(0, color="0.5", lw=0.6)
    style(ax)
    ax.set_ylabel(r"$k\,T_E(k)$  (premultiplied)")
    ax.set_title("KE shell budget")
    ax.legend(fontsize=8)

    ax = axs[2]
    pi_se = np.sqrt(np.cumsum(cut(se["E_adv"]) ** 2))
    ax.semilogx(k, flux(acc["E_adv"]), "C0-", lw=1.8,
                label=r"$\Pi_E(k)$ advective flux")
    ax.fill_between(k, flux(acc["E_adv"]) - pi_se, flux(acc["E_adv"]) + pi_se,
                    color="C0", alpha=0.18, lw=0)
    ax.semilogx(k, np.cumsum(cut(inferred)), "C1-", lw=1.4,
                label=r"$\int_0^k$ stretching input (closure-inferred)")
    ax.semilogx(k, -np.cumsum(cut(acc["E_diss"])), "C2-", lw=1.4,
                label=r"$-\int_0^k$ dissipation")
    ax.axhline(0, color="0.5", lw=0.6)
    style(ax)
    ax.set_ylabel("cumulative rate")
    ax.set_title(r"KE flux: $\Pi_E<0$ = inverse cascade")
    ax.legend(fontsize=8)

    fig.suptitle("Vorticity-equation ENERGY budget, radial shells, ghost-projected", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "energy_budget.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # =========================== ENSTROPHY ==============================
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axs[0]
    Z = cut(acc["Z_spec"])
    ax.loglog(k, Z, "k-", lw=1.8, label=r"$\mathcal{Z}(k)$")
    style(ax)
    ax.set_ylabel(r"$\mathcal{Z}(k)$")
    ax.set_title("Enstrophy spectrum")
    ax.legend(fontsize=8)

    ax = axs[1]
    for nm, lab, c in (("Z_adv", "advection (transfer)", "C0"),
                       ("Z_str", r"stretching $\Re\langle q^*\partial_Z w\rangle$", "C3"),
                       ("Z_diss", r"dissipation $-\nu k^2$", "C2")):
        ax.semilogx(k, k * cut(acc[nm]), color=c, lw=1.5, label=lab)
        ax.fill_between(k, k * cut(acc[nm] - se[nm]), k * cut(acc[nm] + se[nm]),
                        color=c, alpha=0.18, lw=0)
    tot = cut(acc["Z_adv"] + acc["Z_str"] + acc["Z_diss"])
    ax.semilogx(k, k * tot, "k--", lw=1.0, label="sum (net tendency)")
    ax.axhline(0, color="0.5", lw=0.6)
    style(ax)
    ax.set_ylabel(r"$k\,T_Z(k)$  (premultiplied)")
    ax.set_title("Enstrophy shell budget")
    ax.legend(fontsize=8)

    ax = axs[2]
    piz_se = np.sqrt(np.cumsum(cut(se["Z_adv"]) ** 2))
    ax.semilogx(k, flux(acc["Z_adv"]), "C0-", lw=1.8,
                label=r"$\Pi_Z(k)$ advective flux")
    ax.fill_between(k, flux(acc["Z_adv"]) - piz_se, flux(acc["Z_adv"]) + piz_se,
                    color="C0", alpha=0.18, lw=0)
    ax.semilogx(k, np.cumsum(cut(acc["Z_str"])), "C3-", lw=1.4,
                label=r"$\int_0^k$ stretching input")
    ax.semilogx(k, -np.cumsum(cut(acc["Z_diss"])), "C2-", lw=1.4,
                label=r"$-\int_0^k$ dissipation")
    ax.axhline(0, color="0.5", lw=0.6)
    style(ax)
    ax.set_ylabel("cumulative rate")
    ax.set_title(r"Enstrophy flux: $\Pi_Z>0$ = forward cascade")
    ax.legend(fontsize=8)

    fig.suptitle("Vorticity-equation ENSTROPHY budget, radial shells, ghost-projected", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "enstrophy_budget.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # =========================== BAROTROPIC =============================
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axs[0]
    ax.semilogx(k, k * cut(acc["Ebt_adv"]), "C0-", lw=1.6,
                label="advective transfer into bt mode")
    ax.fill_between(k, k * cut(acc["Ebt_adv"] - se["Ebt_adv"]),
                    k * cut(acc["Ebt_adv"] + se["Ebt_adv"]),
                    color="C0", alpha=0.18, lw=0)
    ax.semilogx(k, k * cut(acc["Ebt_diss"]), "C2-", lw=1.4, label="bt dissipation")
    ax.semilogx(k, k * cut(acc["Ebt_adv"] + acc["Ebt_diss"]), "k--", lw=1.0, label="sum")
    ax.axhline(0, color="0.5", lw=0.6)
    style(ax)
    ax.set_ylabel(r"$k\,T_{E_{bt}}(k)$")
    ax.set_title("Barotropic KE budget\n(stretching $\\equiv 0$: $w$ vanishes at walls)")
    ax.legend(fontsize=8)

    ax = axs[1]
    pib_se = np.sqrt(np.cumsum(cut(se["Ebt_adv"]) ** 2))
    ax.semilogx(k, flux(acc["Ebt_adv"]), "C0-", lw=1.8, label=r"$\Pi_{E_{bt}}(k)$")
    ax.fill_between(k, flux(acc["Ebt_adv"]) - pib_se, flux(acc["Ebt_adv"]) + pib_se,
                    color="C0", alpha=0.18, lw=0)
    ax.axhline(0, color="0.5", lw=0.6)
    style(ax)
    ax.set_ylabel("cumulative rate")
    ax.set_title("Barotropic KE flux (the LSV pipeline)")
    ax.legend(fontsize=8)

    ax = axs[2]
    ts_t = np.array(series["t"])
    ax.plot(ts_t, np.array(series["E_bt"]), "C0-", lw=1.2, label=r"$E_{bt}$")
    ax.plot(ts_t, np.array(series["E_tot"]), "k-", lw=1.0, alpha=0.7, label=r"$E$ total")
    ax2 = ax.twinx()
    ax2.plot(ts_t, np.array(series["E_str_sum"]), "C3-", lw=0.7, alpha=0.6)
    ax2.set_ylabel(r"$\sum_k T_{E,str}$ (osc.)", color="C3", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="C3", labelsize=7)
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("energy")
    ax.set_title("Stationarity & conversion-term oscillation")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Barotropic projection: nonlinear baroclinic$\\to$barotropic transfer", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "barotropic_budget.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
