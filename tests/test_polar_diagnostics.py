"""Gate tests for nhqg/polar_diagnostics.py (lead-written, READ-ONLY for executors).

API contract under test (all pure numpy; field convention field[ix, iy] with
x = ix*L/Nx, y = iy*L/Nx, periodic; center defaults to (L/2, L/2)):

  polar_resample(field, L, r_max, n_r=64, n_theta=256, center=None)
      -> (r (n_r,), theta (n_theta,), F (n_r, n_theta))
      r_i = (i+0.5)*r_max/n_r, theta_j = 2*pi*j/n_theta, bilinear interp
      with periodic wrap.
  azimuthal_energy_spectrum(field, L, r_max, m_max=32, n_r=64, n_theta=None,
                            center=None) -> (m (m_max+1,), E_m (m_max+1,))
      n_theta default max(256, 8*m_max); per-ring rfft over theta normalized
      by n_theta; E_m = sum_i r_i * dr * c_m * |f_m(r_i)|^2, c_0=1, c_m=2
      for m>=1.
  radial_profile(field, L, r_max, n_r=64, n_theta=256, center=None)
      -> (r, prof) with prof the azimuthal mean per ring.
  vortex_positions(zeta, L, threshold_frac=0.5, min_separation=None)
      -> (n, 2) array of (x, y); local maxima above threshold_frac*max,
      strictly greater than all 8 periodic neighbors, greedy non-max
      suppression at min_separation (default L/32, periodic metric),
      3-point parabolic subpixel refinement per axis (offsets clipped to
      [-0.5, 0.5] cells); empty (0, 2) if max(zeta) <= 0.
  trap_mask(Nx, L, r_star, center=None) -> (Nx, Nx) float64, 1.0 where
      r < r_star else 0.0.
"""

import numpy as np
import pytest

from nhqg.polar_diagnostics import (
    polar_resample, azimuthal_energy_spectrum, radial_profile,
    vortex_positions, trap_mask,
)

L = 20.0
NX = 256


def _grid():
    xy = np.arange(NX) * (L / NX)
    X, Y = np.meshgrid(xy, xy, indexing='ij')
    return X, Y


def _polar(X, Y, cx=L / 2, cy=L / 2):
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    th = np.arctan2(Y - cy, X - cx)
    return r, th


def _periodic_gauss(X, Y, x0, y0, width):
    dx = ((X - x0 + L / 2) % L) - L / 2
    dy = ((Y - y0 + L / 2) % L) - L / 2
    return np.exp(-(dx ** 2 + dy ** 2) / width ** 2)


class TestResample:
    def test_shapes_and_axes(self):
        X, Y = _grid()
        r, th, F = polar_resample(X * 0 + 1.0, L, r_max=8.0, n_r=32, n_theta=128)
        assert r.shape == (32,) and th.shape == (128,) and F.shape == (32, 128)
        assert abs(r[0] - 0.5 * 8.0 / 32) < 1e-12
        assert abs(th[1] - 2 * np.pi / 128) < 1e-12
        assert np.max(np.abs(F - 1.0)) < 1e-12   # constant field stays constant


class TestAzimuthalSpectrum:
    def test_m5_ring_pattern(self):
        X, Y = _grid()
        r, th = _polar(X, Y)
        f = np.exp(-((r - 4.0) / 0.8) ** 2) * np.cos(5 * th)
        m, E = azimuthal_energy_spectrum(f, L, r_max=8.0, m_max=16)
        assert m.shape == (17,) and E.shape == (17,)
        assert int(np.argmax(E)) == 5
        assert E[5] >= 0.9 * E.sum()

    def test_axisymmetric_goes_to_m0(self):
        X, Y = _grid()
        r, _ = _polar(X, Y)
        f = np.exp(-((r - 4.0) / 0.8) ** 2)
        _, E = azimuthal_energy_spectrum(f, L, r_max=8.0, m_max=16)
        assert int(np.argmax(E)) == 0
        assert E[0] >= 0.99 * E.sum()


class TestRadialProfile:
    def test_ring_peak_location(self):
        X, Y = _grid()
        r, _ = _polar(X, Y)
        f = np.exp(-((r - 4.0) / 0.8) ** 2)
        rr, prof = radial_profile(f, L, r_max=8.0, n_r=64)
        dr = 8.0 / 64
        assert abs(rr[int(np.argmax(prof))] - 4.0) <= dr


class TestVortexPositions:
    def test_three_gaussians_recovered(self):
        X, Y = _grid()
        truth = np.array([[6.0, 6.0], [14.0, 10.0], [8.0, 15.0]])
        z = sum(_periodic_gauss(X, Y, x0, y0, 0.7) for x0, y0 in truth)
        pos = vortex_positions(z, L, threshold_frac=0.5)
        assert pos.shape == (3, 2)
        for x0, y0 in truth:
            d = np.sqrt(((pos[:, 0] - x0 + L / 2) % L - L / 2) ** 2
                        + ((pos[:, 1] - y0 + L / 2) % L - L / 2) ** 2)
            assert d.min() < 0.15

    def test_periodic_wrap(self):
        X, Y = _grid()
        z = _periodic_gauss(X, Y, 0.2, 19.8, 0.7)
        pos = vortex_positions(z, L, threshold_frac=0.5)
        assert pos.shape == (1, 2)
        d = np.sqrt(((pos[0, 0] - 0.2 + L / 2) % L - L / 2) ** 2
                    + ((pos[0, 1] - 19.8 + L / 2) % L - L / 2) ** 2)
        assert d < 0.15

    def test_empty_when_nonpositive(self):
        z = -np.ones((NX, NX))
        pos = vortex_positions(z, L)
        assert pos.shape == (0, 2)


class TestTrapMask:
    def test_area_fraction(self):
        m = trap_mask(NX, L, r_star=4.0)
        assert m.shape == (NX, NX) and m.dtype == np.float64
        frac = m.mean()
        expected = np.pi * 4.0 ** 2 / L ** 2
        assert abs(frac - expected) < 0.02 * expected
