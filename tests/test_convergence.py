"""Tests verifying second-order spatial convergence of the finite difference scheme."""

import pytest
import numpy as np

from neutron_diffusion.analytics import analytical_fundamental_flux, analytical_keff
from neutron_diffusion.config import get_baseline_config
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.metrics import compute_flux_errors, estimate_convergence_order
from neutron_diffusion.solver import PowerIterationSolver, SolverConfig


def test_spatial_convergence_order_keff_and_flux() -> None:
    """Verify that numerical discretization exhibits ~O(dx^2) convergence."""
    base_cfg = get_baseline_config()
    geom = base_cfg.to_geometry()
    mats = base_cfg.to_materials()
    ana_keff = analytical_keff(mats, geom, mode=1)

    mesh_cells = [20, 40, 80, 160]
    dx_vals = []
    keff_errors = []
    flux_errors = []

    for n in mesh_cells:
        mesh = UniformMesh1D(geometry=geom, num_cells=n)
        cfg = SolverConfig(tolerance_k=1.0e-11, tolerance_flux=1.0e-10, max_iterations=2000)
        solver = PowerIterationSolver(mesh=mesh, materials=mats, config=cfg)
        res = solver.solve()

        ana_flux = analytical_fundamental_flux(mesh, normalization="l2")
        f_err = compute_flux_errors(res.flux, ana_flux, mesh)

        dx_vals.append(mesh.dx)
        keff_errors.append(abs(res.keff - ana_keff) / ana_keff)
        flux_errors.append(f_err["l2_relative_error"])

    # 1. Monotonic decrease in eigenvalue error with mesh refinement
    for i in range(len(keff_errors) - 1):
        assert keff_errors[i + 1] < keff_errors[i], "k_eff error must strictly decrease upon mesh refinement"

    # 2. Observed order of convergence calculation for eigenvalue
    keff_order, keff_r2 = estimate_convergence_order(dx_vals, keff_errors)

    # Theoretical expectation is second-order (p = 2.0)
    assert 1.98 <= keff_order <= 2.02, f"Expected k_eff order ~ 2.0, got {keff_order:.3f}"
    assert keff_r2 > 0.9999, f"Expected near-perfect linearity in log-log fit, got R^2 = {keff_r2:.4f}"

    # 3. Flux shape fidelity: Central difference discretization preserves exact discrete sine mode,
    # so nodal flux error is bounded by solver convergence tolerance (~1e-10)
    for f_err_val in flux_errors:
        assert f_err_val < 1.0e-8, f"Flux error should be bounded by solver tolerance, got {f_err_val:.2e}"
