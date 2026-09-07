"""Reproducible scientific experiments: baseline verification, mesh convergence, and parameter sensitivity."""

import csv
import json
from pathlib import Path
from typing import Any
import numpy as np

from neutron_diffusion.analytics import (
    analytical_buckling,
    analytical_fundamental_flux,
    analytical_keff,
)
from neutron_diffusion.config import SimulationConfig, get_baseline_config
from neutron_diffusion.metrics import (
    compute_flux_errors,
    compute_keff_errors,
    estimate_convergence_order,
)
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.plotting import (
    plot_convergence_history,
    plot_flux_comparison,
    plot_flux_sensitivity,
    plot_mesh_convergence,
    plot_parameter_sensitivity,
    plot_power_distribution,
)
from neutron_diffusion.solver import PowerIterationSolver


def run_baseline_experiment(
    config: SimulationConfig | None = None,
    output_dir: str | Path = "results",
) -> dict[str, Any]:
    """Execute the baseline verification simulation, produce publication plots, and save metrics.

    Parameters
    ----------
    config : SimulationConfig, optional
        Simulation parameters. Defaults to standard baseline configuration.
    output_dir : str or Path, optional
        Root results directory. Default is "results".

    Returns
    -------
    dict[str, Any]
        Summary dictionary containing parameters, convergence history, and error metrics.
    """
    cfg = config or get_baseline_config()
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    metric_dir = out_path / "metrics"
    table_dir = out_path / "tables"

    for d in (fig_dir, metric_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    geom = cfg.to_geometry()
    mats = cfg.to_materials()
    mesh = cfg.to_mesh()
    solver_cfg = cfg.to_solver_config()

    # 1. Run numerical solver
    solver = PowerIterationSolver(mesh=mesh, materials=mats, config=solver_cfg)
    result = solver.solve()

    # 2. Compute analytical reference
    ana_keff = analytical_keff(mats, geom, mode=1)
    ana_flux = analytical_fundamental_flux(mesh, normalization=cfg.flux_normalization)
    bg_sq = analytical_buckling(geom, mode=1)

    # 3. Error analysis
    keff_errs = compute_keff_errors(result.keff, ana_keff)
    flux_errs = compute_flux_errors(result.flux, ana_flux, mesh)

    # 4. Generate publication plots
    plot_flux_comparison(
        mesh=mesh,
        numerical_flux=result.flux,
        analytical_flux=ana_flux,
        output_path=fig_dir / "flux_comparison.png",
    )
    plot_power_distribution(
        mesh=mesh,
        power=result.normalized_power,
        output_path=fig_dir / "power_distribution.png",
        analytical_flux_shape=ana_flux,
    )
    plot_convergence_history(
        result=result,
        output_path=fig_dir / "power_iteration_convergence.png",
    )

    # 5. Export machine-readable summary
    summary: dict[str, Any] = {
        "benchmark": "1D Homogeneous Slab Reactor Baseline",
        "parameters": {
            "length_cm": geom.length,
            "diffusion_coefficient_cm": mats.D,
            "absorption_cross_section_cm_inv": mats.sigma_a,
            "fission_production_cross_section_cm_inv": mats.nu_sigma_f,
            "geometric_buckling_cm_inv_sq": bg_sq,
            "diffusion_length_cm": mats.diffusion_length,
            "k_infinity": mats.k_inf,
        },
        "discretization": {
            "num_cells": mesh.num_cells,
            "num_interior_nodes": mesh.num_interior,
            "dx_cm": mesh.dx,
        },
        "convergence": {
            "converged": result.converged,
            "iterations": result.iterations,
            "tolerance_k": solver_cfg.tolerance_k,
            "tolerance_flux": solver_cfg.tolerance_flux,
            "runtime_seconds": result.runtime_seconds,
        },
        "results": {
            "numerical_keff": result.keff,
            "analytical_keff": ana_keff,
            "absolute_error": keff_errs["absolute_error"],
            "relative_error": keff_errs["relative_error"],
            "pcm_error": keff_errs["pcm_error"],
            "flux_l2_relative_error": flux_errs["l2_relative_error"],
            "flux_linf_error": flux_errs["linf_error"],
        },
    }

    summary_file = metric_dir / "baseline_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Export flux and power table
    csv_file = table_dir / "baseline_flux.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_cm", "numerical_flux", "analytical_flux", "residual", "normalized_power"])
        for xi, n_phi, a_phi, pwr in zip(mesh.x, result.flux, ana_flux, result.normalized_power):
            writer.writerow([f"{xi:.6f}", f"{n_phi:.8e}", f"{a_phi:.8e}", f"{n_phi - a_phi:.8e}", f"{pwr:.8e}"])

    return summary


def run_mesh_convergence_study(
    mesh_cells_list: list[int] | None = None,
    output_dir: str | Path = "results",
) -> dict[str, Any]:
    """Execute systematic spatial mesh-refinement study to measure observed order of convergence.

    Parameters
    ----------
    mesh_cells_list : list of int, optional
        Sequence of spatial cell counts. Defaults to [20, 40, 80, 160, 320, 640].
    output_dir : str or Path, optional
        Root results directory. Default is "results".

    Returns
    -------
    dict[str, Any]
        Mesh refinement dataset including observed orders for eigenvalue and eigenflux.
    """
    cells_list = mesh_cells_list or [20, 40, 80, 160, 320, 640]
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    metric_dir = out_path / "metrics"
    table_dir = out_path / "tables"

    for d in (fig_dir, metric_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    base_cfg = get_baseline_config()
    geom = base_cfg.to_geometry()
    mats = base_cfg.to_materials()
    ana_keff = analytical_keff(mats, geom, mode=1)

    dx_values: list[float] = []
    keff_numerical: list[float] = []
    keff_rel_errors: list[float] = []
    flux_l2_errors: list[float] = []
    runtimes: list[float] = []
    iterations_list: list[int] = []

    for n_cells in cells_list:
        mesh = UniformMesh1D(geometry=geom, num_cells=n_cells)
        # Tight tolerances for high precision error evaluation
        solver_cfg = base_cfg.to_solver_config()
        solver_cfg.tolerance_k = 1.0e-11
        solver_cfg.tolerance_flux = 1.0e-10

        solver = PowerIterationSolver(mesh=mesh, materials=mats, config=solver_cfg)
        res = solver.solve()

        ana_flux = analytical_fundamental_flux(mesh, normalization=base_cfg.flux_normalization)
        k_err = compute_keff_errors(res.keff, ana_keff)
        f_err = compute_flux_errors(res.flux, ana_flux, mesh)

        dx_values.append(mesh.dx)
        keff_numerical.append(res.keff)
        keff_rel_errors.append(k_err["relative_error"])
        flux_l2_errors.append(f_err["l2_relative_error"])
        runtimes.append(res.runtime_seconds)
        iterations_list.append(res.iterations)

    # Estimate observed asymptotic convergence order p: Error ~ C * (dx)^p
    keff_order, keff_r2 = estimate_convergence_order(dx_values, keff_rel_errors)
    flux_order, flux_r2 = estimate_convergence_order(dx_values, flux_l2_errors)

    # Generate log-log mesh convergence figure
    plot_mesh_convergence(
        dx_list=dx_values,
        keff_rel_errors=keff_rel_errors,
        flux_l2_errors=flux_l2_errors,
        output_path=fig_dir / "mesh_convergence.png",
        keff_order=keff_order,
        flux_order=flux_order,
    )

    study_data: dict[str, Any] = {
        "analytical_keff": ana_keff,
        "mesh_refinements": [
            {
                "num_cells": int(n),
                "dx_cm": float(dx),
                "numerical_keff": float(k),
                "relative_keff_error": float(rel_k),
                "flux_l2_relative_error": float(l2_f),
                "iterations": int(it),
                "runtime_seconds": float(rt),
            }
            for n, dx, k, rel_k, l2_f, it, rt in zip(
                cells_list, dx_values, keff_numerical, keff_rel_errors, flux_l2_errors, iterations_list, runtimes
            )
        ],
        "observed_convergence": {
            "keff_convergence_order": keff_order,
            "keff_r_squared": keff_r2,
            "flux_convergence_order": flux_order,
            "flux_r_squared": flux_r2,
            "expected_theoretical_order": 2.0,
        },
    }

    with open(metric_dir / "mesh_convergence.json", "w", encoding="utf-8") as f:
        json.dump(study_data, f, indent=2)

    # CSV Table
    with open(table_dir / "mesh_convergence.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cells", "dx_cm", "keff_numerical", "rel_error_keff", "flux_l2_error", "iterations", "runtime_s"])
        for row in study_data["mesh_refinements"]:
            writer.writerow([
                row["num_cells"],
                f"{row['dx_cm']:.6e}",
                f"{row['numerical_keff']:.10f}",
                f"{row['relative_keff_error']:.6e}",
                f"{row['flux_l2_relative_error']:.6e}",
                row["iterations"],
                f"{row['runtime_seconds']:.6f}",
            ])

    return study_data


def run_parameter_study(output_dir: str | Path = "results") -> dict[str, Any]:
    """Conduct reactor physics parameter sensitivity sweeps across L, Sigma_a, D, and nuSigma_f.

    Parameters
    ----------
    output_dir : str or Path, optional
        Root results directory. Default is "results".

    Returns
    -------
    dict[str, Any]
        Dictionary storing parameter values and corresponding k_eff results.
    """
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    metric_dir = out_path / "metrics"
    table_dir = out_path / "tables"

    for d in (fig_dir, metric_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    base = get_baseline_config()
    solver_cfg = base.to_solver_config()

    # Parameter sweeps
    l_vals = [30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0]
    sa_vals = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035]
    d_vals = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    nf_vals = [0.018, 0.020, 0.022, 0.025, 0.028, 0.032]

    results_data: dict[str, dict[str, list[float]]] = {
        "length": {"param_values": l_vals, "keff_values": []},
        "sigma_a": {"param_values": sa_vals, "keff_values": []},
        "D": {"param_values": d_vals, "keff_values": []},
        "nu_sigma_f": {"param_values": nf_vals, "keff_values": []},
    }

    # 1. Sweep Slab Length L
    for l_val in l_vals:
        cfg = SimulationConfig(length=l_val, D=base.D, sigma_a=base.sigma_a, nu_sigma_f=base.nu_sigma_f, num_cells=100)
        sol = PowerIterationSolver(mesh=cfg.to_mesh(), materials=cfg.to_materials(), config=solver_cfg).solve()
        results_data["length"]["keff_values"].append(sol.keff)

    # 2. Sweep Absorption Cross Section Sigma_a
    for sa_val in sa_vals:
        cfg = SimulationConfig(length=base.length, D=base.D, sigma_a=sa_val, nu_sigma_f=base.nu_sigma_f, num_cells=100)
        sol = PowerIterationSolver(mesh=cfg.to_mesh(), materials=cfg.to_materials(), config=solver_cfg).solve()
        results_data["sigma_a"]["keff_values"].append(sol.keff)

    # 3. Sweep Diffusion Coefficient D
    for d_val in d_vals:
        cfg = SimulationConfig(length=base.length, D=d_val, sigma_a=base.sigma_a, nu_sigma_f=base.nu_sigma_f, num_cells=100)
        sol = PowerIterationSolver(mesh=cfg.to_mesh(), materials=cfg.to_materials(), config=solver_cfg).solve()
        results_data["D"]["keff_values"].append(sol.keff)

    # 4. Sweep Fission Production Cross Section nuSigma_f
    for nf_val in nf_vals:
        cfg = SimulationConfig(length=base.length, D=base.D, sigma_a=base.sigma_a, nu_sigma_f=nf_val, num_cells=100)
        sol = PowerIterationSolver(mesh=cfg.to_mesh(), materials=cfg.to_materials(), config=solver_cfg).solve()
        results_data["nu_sigma_f"]["keff_values"].append(sol.keff)

    # 5. Flux profile sensitivity across varying core dimensions
    flux_profiles = []
    for l_sub in [50.0, 100.0, 200.0]:
        cfg = SimulationConfig(length=l_sub, D=base.D, sigma_a=base.sigma_a, nu_sigma_f=base.nu_sigma_f, num_cells=100)
        m = cfg.to_mesh()
        sol = PowerIterationSolver(mesh=m, materials=cfg.to_materials(), config=solver_cfg).solve()
        flux_profiles.append((l_sub, m.x, sol.flux))

    # Generate figures
    base_ana_keff = analytical_keff(base.to_materials(), base.to_geometry(), mode=1)
    plot_parameter_sensitivity(results_data, base_ana_keff, fig_dir / "parameter_sensitivity_keff.png")
    plot_flux_sensitivity(flux_profiles, fig_dir / "flux_sensitivity.png")

    with open(metric_dir / "parameter_study.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    with open(table_dir / "parameter_study.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "keff"])
        for param_name, data in results_data.items():
            for val, k in zip(data["param_values"], data["keff_values"]):
                writer.writerow([param_name, val, f"{k:.8f}"])

    return results_data
