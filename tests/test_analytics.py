"""Tests for analytical reference solutions and eigenvalue formulas."""

import math
import pytest
import numpy as np

from neutron_diffusion.analytics import (
    analytical_buckling,
    analytical_critical_production,
    analytical_fundamental_flux,
    analytical_keff,
)
from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D


def test_analytical_buckling() -> None:
    geom = SlabGeometry(length=100.0)
    bg_sq = analytical_buckling(geom, mode=1)
    expected_bg_sq = (math.pi / 100.0) ** 2
    assert pytest.approx(expected_bg_sq, rel=1e-12) == bg_sq
    assert bg_sq > 0.0

    # Mode 2 buckling: (2*pi/L)^2 = 4 * Bg1^2
    bg2_sq = analytical_buckling(geom, mode=2)
    assert pytest.approx(4.0 * expected_bg_sq, rel=1e-12) == bg2_sq

    with pytest.raises(ValueError, match="Harmonic mode must be >= 1"):
        analytical_buckling(geom, mode=0)


def test_analytical_keff_baseline() -> None:
    """Verify single-fraction formula: k_eff = nuSigma_f / (Sigma_a + D * Bg^2)."""
    geom = SlabGeometry(length=100.0)
    mats = MaterialProperties(D=1.0, sigma_a=0.02, nu_sigma_f=0.025)

    bg_sq = (math.pi / 100.0) ** 2
    expected_keff = 0.025 / (0.02 + 1.0 * bg_sq)
    assert pytest.approx(1.1912160756, abs=1e-6) == expected_keff

    calculated_keff = analytical_keff(mats, geom, mode=1)
    assert pytest.approx(expected_keff, rel=1e-12) == calculated_keff

    # Verify thermal diffusion length: L_d = sqrt(D / Sigma_a)
    expected_ld = math.sqrt(1.0 / 0.02)
    assert pytest.approx(7.0710678118654755, rel=1e-12) == expected_ld
    assert pytest.approx(expected_ld, rel=1e-12) == mats.diffusion_length

    # Verify infinite multiplication factor: k_inf = nuSigma_f / Sigma_a
    assert pytest.approx(1.25, rel=1e-12) == mats.k_inf


def test_analytical_keff_critical_consistency() -> None:
    """Verify that setting nuSigma_f = Sigma_a + D * Bg^2 yields exact k_eff = 1.0."""
    geom = SlabGeometry(length=100.0)
    base_mats = MaterialProperties(D=1.0, sigma_a=0.02, nu_sigma_f=0.025)

    crit_nu_sigma_f = analytical_critical_production(base_mats, geom)
    crit_mats = MaterialProperties(D=1.0, sigma_a=0.02, nu_sigma_f=crit_nu_sigma_f)

    keff_crit = analytical_keff(crit_mats, geom, mode=1)
    assert pytest.approx(1.0, rel=1e-14) == keff_crit
    assert pytest.approx(0.02098696044, abs=1e-8) == crit_nu_sigma_f


def test_analytical_flux_properties() -> None:
    geom = SlabGeometry(length=100.0)
    mesh = UniformMesh1D(geometry=geom, num_cells=100)

    # Test L2 normalization
    flux_l2 = analytical_fundamental_flux(mesh, normalization="l2")
    assert len(flux_l2) == mesh.num_points
    assert flux_l2[0] == pytest.approx(0.0, abs=1e-14)
    assert flux_l2[-1] == pytest.approx(0.0, abs=1e-14)
    assert np.all(flux_l2[1:-1] > 0.0)  # Strictly positive in interior

    # Symmetry about core center x = L/2
    assert pytest.approx(flux_l2, rel=1e-12) == flux_l2[::-1]

    # Continuous integral of squared flux should be exactly 1.0
    integral_sq = mesh.integrate(flux_l2 ** 2)
    assert pytest.approx(1.0, rel=1e-4) == integral_sq

    # Test max normalization
    flux_max = analytical_fundamental_flux(mesh, normalization="max")
    assert np.max(flux_max) == pytest.approx(1.0, rel=1e-12)
    assert flux_max[len(flux_max) // 2] == pytest.approx(1.0, rel=1e-12)

    # Test integral normalization
    flux_int = analytical_fundamental_flux(mesh, normalization="integral")
    assert mesh.integrate(flux_int) == pytest.approx(1.0, rel=1e-12)

    with pytest.raises(ValueError, match="Unknown normalization"):
        analytical_fundamental_flux(mesh, normalization="invalid_norm")
