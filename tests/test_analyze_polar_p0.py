"""Lead-written acceptance gate for scripts/analyze_polar_p0.py (P0 analysis).

(Authored under scripts/ during the concurrent M2b delegation so the
`pytest tests/` contract count stayed fixed; folded into tests/ at commit.)

Oracle: a synthetic 19-vortex lattice (1 center + ring of 6 at r=10 +
ring of 12 at r=20, Gaussian cores) with exactly known geometry:
  R_true = sqrt((0 + 6*10^2 + 12*20^2)/19) = 16.8585...
  median NN spacing ~ 10 (center-ring chords and ring-ring gaps all ~10)
"""

import importlib.util
from pathlib import Path

import numpy as np

_SPEC_PATH = Path(__file__).parent.parent / "scripts" / "analyze_polar_p0.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_polar_p0", _SPEC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_lattice(L=200.0, Nx=512, sig=2.0):
    x = np.arange(Nx) * L / Nx
    X, Y = np.meshgrid(x, x, indexing="ij")   # field[ix, iy], x = ix*L/Nx
    c = L / 2.0
    centers = [(c, c)]
    centers += [(c + 10.0 * np.cos(2 * np.pi * j / 6),
                 c + 10.0 * np.sin(2 * np.pi * j / 6)) for j in range(6)]
    centers += [(c + 20.0 * np.cos(2 * np.pi * j / 12 + 0.1),
                 c + 20.0 * np.sin(2 * np.pi * j / 12 + 0.1)) for j in range(12)]
    zeta = np.zeros((Nx, Nx))
    for (cx, cy) in centers:
        dx = (X - cx + L / 2.0) % L - L / 2.0
        dy = (Y - cy + L / 2.0) % L - L / 2.0
        zeta += np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sig ** 2))
    return zeta, L


class TestCrystalMetricsGate:

    def test_synthetic_lattice_recovery(self):
        mod = _load_module()
        zeta, L = _synthetic_lattice()
        m = mod.crystal_metrics(zeta, L, threshold_frac=0.3, min_separation=5.0)
        R_true = np.sqrt((6 * 10.0 ** 2 + 12 * 20.0 ** 2) / 19.0)
        assert m["n_vortices"] == 19
        assert abs(m["R_crystal"] - R_true) < 0.02 * R_true
        assert 9.0 < m["spacing"] < 11.5
        assert m["positions"].shape == (19, 2)

    def test_noise_robustness(self):
        mod = _load_module()
        zeta, L = _synthetic_lattice()
        rng = np.random.default_rng(1)
        zeta_noisy = zeta + 0.02 * rng.standard_normal(zeta.shape)
        m = mod.crystal_metrics(zeta_noisy, L, threshold_frac=0.3,
                                min_separation=5.0)
        R_true = np.sqrt((6 * 10.0 ** 2 + 12 * 20.0 ** 2) / 19.0)
        assert m["n_vortices"] == 19
        assert abs(m["R_crystal"] - R_true) < 0.03 * R_true

    def test_empty_field(self):
        mod = _load_module()
        zeta = np.full((64, 64), -1.0)   # no positive vorticity anywhere
        m = mod.crystal_metrics(zeta, 100.0)
        assert m["n_vortices"] == 0
        assert np.isnan(m["spacing"])
