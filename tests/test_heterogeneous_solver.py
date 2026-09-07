"""Tests for multi-region heterogeneous power iteration solver and interface continuity."""

import pytest
import numpy as np

from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.regions import (
    MaterialRegion,
    PiecewiseMaterialModel,
    create_reflected_slab,
)
from neutron_diffusion.heterogeneous_solver import HeterogeneousPowerIterationSolver
from neutron_diffusion.solver import PowerIterationSolver, SolverConfig


def test_heterogeneous_solver_homogeneous_equivalence() -> None:
    """Verify that HeterogeneousPowerIterationSolver produces identical k_eff to
    PowerIterationSolver when material regions are identical."""
    geom = SlabGeometry(length=100.0)
    mats = MaterialProperties(D=1.0, sigma_a=0.02, nu_sigma_f=0.025)
    mesh = UniformMesh1D(geometry=geom, num_cells=50)

    # 1. Homogeneous solver
    cfg = SolverConfig(tolerance_k=1.0e-9, tolerance_flux=1.0e-8)
    hom_res = PowerIterationSolver(mesh=mesh, materials=mats, config=cfg).solve()

    # 2. Heterogeneous solver with identical regions
    r1 = MaterialRegion(name="r1", x_min=0.0, x_max=40.0, materials=mats)
    r2 = MaterialRegion(name="r2", x_min=40.0, x_max=100.0, materials=mats)
    model = PiecewiseMaterialModel(geometry=geom, regions=(r1, r2))

    het_res = HeterogeneousPowerIterationSolver(mesh=mesh, material_model=model, config=cfg).solve()

    assert pytest.approx(hom_res.keff, rel=1e-8) == het_res.keff
    assert np.allclose(hom_res.flux, het_res.flux, atol=1e-7)


def test_reflected_core_convergence_and_flux_properties() -> None:
    fuel = MaterialProperties(D=1.20, sigma_a=0.022, nu_sigma_f=0.026)
    refl = MaterialProperties(D=0.85, sigma_a=0.003, nu_sigma_f=0.0)
    # 20 cm refl, 60 cm fuel, 20 cm refl
    model = create_reflected_slab(60.0, 20.0, fuel, refl, symmetric=True)
    mesh = UniformMesh1D(geometry=model.geometry, num_cells=100)

    solver = HeterogeneousPowerIterationSolver(mesh=mesh, material_model=model)
    res = solver.solve()

    assert res.converged
    assert res.keff > 0.0

    # 1. Non-negativity
    assert np.all(res.flux >= 0.0)
    assert np.all(res.flux[1:-1] > 0.0)

    # 2. Symmetry of flux for symmetric reflected slab: phi(x) == phi(L - x)
    assert np.allclose(res.flux, res.flux[::-1], rtol=1e-5, atol=1e-7)

    # 3. Power distribution is zero in reflector
    _, _, _, fiss_mask = model.sample_nodal_properties(mesh)
    assert np.all(res.normalized_power[~fiss_mask] == 0.0)
    assert np.all(res.normalized_power[fiss_mask] > 0.0)


def test_interface_flux_and_current_continuity() -> None:
    """Verify that scalar flux and net current J = -D dphi/dx satisfy physical continuity
    at material interfaces within numerical discretization tolerance."""
    fuel = MaterialProperties(D=1.20, sigma_a=0.022, nu_sigma_f=0.026)
    refl = MaterialProperties(D=0.85, sigma_a=0.003, nu_sigma_f=0.0)
    refl_t = 20.0
    fuel_t = 60.0
    model = create_reflected_slab(fuel_t, refl_t, fuel, refl, symmetric=True)
    # Refined mesh with dx = 0.5 cm
    mesh = UniformMesh1D(geometry=model.geometry, num_cells=200)

    cfg = SolverConfig(tolerance_k=1.0e-9, tolerance_flux=1.0e-8)
    solver = HeterogeneousPowerIterationSolver(mesh=mesh, material_model=model, config=cfg)
    res = solver.solve()

    # Left interface at x = 20.0 cm
    diag_left = res.evaluate_interface_balance(refl_t)

    # Flux continuity check: smooth variation across interface node
    # Since flux is continuous, the difference between node value and neighbor average is O(dx^2)
    assert diag_left["flux_jump"] < 0.01

    # Current continuity check: face currents on either side of interface node
    # Physical net current J = -D dphi/dx is continuous across interface.
    # Discretization error on face currents is O(dx)
    current_mismatch = diag_left["current_difference"]
    assert current_mismatch < 0.05, f"Current mismatch across interface too large: {current_mismatch}"

    # Right interface at x = 80.0 cm
    diag_right = res.evaluate_interface_balance(refl_t + fuel_t)
    assert diag_right["flux_jump"] < 0.01
    assert diag_right["current_difference"] < 0.05
