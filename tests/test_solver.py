"""Tests for power iteration eigenvalue solver and convergence behaviour."""

import pytest
import numpy as np

from neutron_diffusion.analytics import analytical_keff
from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.solver import (
    ConvergenceError,
    PowerIterationSolver,
    SolverConfig,
)


@pytest.fixture
def baseline_simulation() -> tuple[UniformMesh1D, MaterialProperties, SlabGeometry]:
    geom = SlabGeometry(length=100.0)
    mesh = UniformMesh1D(geometry=geom, num_cells=100)
    mats = MaterialProperties(D=1.0, sigma_a=0.02, nu_sigma_f=0.025)
    return mesh, mats, geom


def test_power_iteration_baseline_convergence(baseline_simulation) -> None:
    mesh, mats, geom = baseline_simulation
    config = SolverConfig(tolerance_k=1.0e-8, tolerance_flux=1.0e-7, max_iterations=500)
    solver = PowerIterationSolver(mesh=mesh, materials=mats, config=config)

    result = solver.solve()

    assert result.converged
    assert result.iterations < 500
    assert result.keff > 1.0

    # Check agreement with analytical k_eff
    ana_keff = analytical_keff(mats, geom, mode=1)
    rel_error = abs(result.keff - ana_keff) / ana_keff

    # For N=100, spatial FD discretization error is on order of 1e-4
    assert rel_error < 2.0e-4

    # Pointwise flux properties
    assert result.flux[0] == 0.0
    assert result.flux[-1] == 0.0
    assert np.all(result.flux[1:-1] > 0.0)  # Fundamental mode must be strictly positive

    # Flux symmetry about core center
    assert np.allclose(result.flux, result.flux[::-1], rtol=1e-6)

    # Power distribution check (average power must equal 1.0)
    avg_power = mesh.integrate(result.normalized_power) / geom.length
    assert pytest.approx(1.0, rel=1e-5) == avg_power


def test_solver_different_initial_guesses(baseline_simulation) -> None:
    mesh, mats, _ = baseline_simulation

    cfg_uni = SolverConfig(initial_flux_guess="uniform", tolerance_k=1.0e-9, tolerance_flux=1.0e-8)
    cfg_sin = SolverConfig(initial_flux_guess="sinusoidal", tolerance_k=1.0e-9, tolerance_flux=1.0e-8)

    res_uni = PowerIterationSolver(mesh=mesh, materials=mats, config=cfg_uni).solve()
    res_sin = PowerIterationSolver(mesh=mesh, materials=mats, config=cfg_sin).solve()

    # Both initial guesses must converge to the same unique fundamental eigenvalue
    assert pytest.approx(res_uni.keff, rel=1e-7) == res_sin.keff

    # And identical normalized flux shapes
    assert np.allclose(res_uni.flux, res_sin.flux, atol=1e-6)


def test_solver_max_iterations_exceeded(baseline_simulation) -> None:
    mesh, mats, _ = baseline_simulation
    # Set max_iterations = 2 with very tight tolerance to force non-convergence
    cfg = SolverConfig(max_iterations=2, tolerance_k=1.0e-12, tolerance_flux=1.0e-12)
    solver = PowerIterationSolver(mesh=mesh, materials=mats, config=cfg)

    with pytest.raises(ConvergenceError, match="failed to converge"):
        solver.solve()


def test_solver_config_validation() -> None:
    with pytest.raises(ValueError, match="tolerance_k must be strictly positive"):
        SolverConfig(tolerance_k=-1.0e-5)

    with pytest.raises(ValueError, match="tolerance_flux must be strictly positive"):
        SolverConfig(tolerance_flux=0.0)

    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        SolverConfig(max_iterations=0)
