"""Computational studies for heterogeneous reactor models: reflector savings and critical dimensions."""

import csv
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.interpolate import CubicSpline

from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.plotting import (
    plot_bare_vs_reflector,
    plot_core_reflector_flux,
    plot_critical_fuel_thickness,
    plot_heterogeneous_convergence,
    plot_heterogeneous_material_map,
    plot_reflector_sensitivity,
)
from neutron_diffusion.regions import PiecewiseMaterialModel, create_reflected_slab
from neutron_diffusion.heterogeneous_solver import (
    HeterogeneousPowerIterationSolver,
    HeterogeneousSolverResult,
)
from neutron_diffusion.solver import PowerIterationSolver, SolverConfig


def get_default_heterogeneous_materials() -> tuple[MaterialProperties, MaterialProperties]:
    """Return physically consistent benchmark fuel and reflector material sets."""
    fuel = MaterialProperties(
        D=1.20,
        sigma_a=0.022,
        nu_sigma_f=0.026,
        name="Enriched UO2 / Moderator Core",
    )
    reflector = MaterialProperties(
        D=0.85,
        sigma_a=0.003,
        nu_sigma_f=0.0,  # Pure scattering/absorbing, zero fission
        name="Heavy Water / Graphite Reflector",
    )
    return fuel, reflector


def run_core_reflector_baseline(output_dir: str | Path = "results") -> dict[str, Any]:
    """Execute baseline symmetric reflected core simulation and produce publication artifacts.

    Configuration:
    - Symmetric slab: [Reflector 20 cm | Fuel Core 80 cm | Reflector 20 cm], total L = 120 cm.
    - Mesh: 240 uniform cells (dx = 0.5 cm, resolving interfaces cleanly).
    """
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    metric_dir = out_path / "metrics"
    table_dir = out_path / "tables"

    for d in (fig_dir, metric_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    fuel_mats, refl_mats = get_default_heterogeneous_materials()
    fuel_t = 80.0
    refl_t = 20.0
    model = create_reflected_slab(
        fuel_thickness=fuel_t,
        reflector_thickness=refl_t,
        fuel_materials=fuel_mats,
        reflector_materials=refl_mats,
        symmetric=True,
    )

    mesh = UniformMesh1D(geometry=model.geometry, num_cells=240)
    solver = HeterogeneousPowerIterationSolver(mesh=mesh, material_model=model)
    result = solver.solve()

    # Interface diagnostic balance
    int_left_diag = result.evaluate_interface_balance(refl_t)
    int_right_diag = result.evaluate_interface_balance(refl_t + fuel_t)

    # Plotting
    plot_core_reflector_flux(result, fig_dir / "core_reflector_flux.png")
    plot_heterogeneous_material_map(model, mesh, fig_dir / "heterogeneous_material_map.png")

    summary: dict[str, Any] = {
        "model": "1D Symmetric Reflected Core (Version 2)",
        "geometry": {
            "total_length_cm": model.geometry.length,
            "fuel_thickness_cm": fuel_t,
            "reflector_thickness_cm": refl_t,
            "interfaces_cm": model.interface_locations,
        },
        "fuel_properties": {
            "D_cm": fuel_mats.D,
            "sigma_a_cm_inv": fuel_mats.sigma_a,
            "nu_sigma_f_cm_inv": fuel_mats.nu_sigma_f,
        },
        "reflector_properties": {
            "D_cm": refl_mats.D,
            "sigma_a_cm_inv": refl_mats.sigma_a,
            "nu_sigma_f_cm_inv": refl_mats.nu_sigma_f,
        },
        "results": {
            "keff": result.keff,
            "iterations": result.iterations,
            "runtime_seconds": result.runtime_seconds,
            "peak_to_average_core_power": float(np.max(result.normalized_power)),
        },
        "interface_diagnostics": {
            "left_interface": int_left_diag,
            "right_interface": int_right_diag,
        },
    }

    with open(metric_dir / "heterogeneous_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save CSV table
    with open(table_dir / "core_reflector_flux.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_cm", "flux", "normalized_power", "region"])
        for xi, flx, pwr in zip(mesh.x, result.flux, result.normalized_power):
            reg = model.find_region_at(xi)
            writer.writerow([f"{xi:.6f}", f"{flx:.8e}", f"{pwr:.8e}", reg.name])

    return summary


def compare_bare_vs_reflected(output_dir: str | Path = "results") -> dict[str, Any]:
    """Execute controlled comparison between identical fuel slab with and without reflector."""
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    metric_dir = out_path / "metrics"
    table_dir = out_path / "tables"

    for d in (fig_dir, metric_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    fuel_mats, refl_mats = get_default_heterogeneous_materials()
    fuel_t = 80.0
    refl_t = 20.0

    # 1. Bare core simulation (length = fuel_t = 80 cm)
    geom_bare = SlabGeometry(length=fuel_t)
    mesh_bare = UniformMesh1D(geometry=geom_bare, num_cells=160)
    bare_solver = PowerIterationSolver(mesh=mesh_bare, materials=fuel_mats)
    bare_result = bare_solver.solve()

    # 2. Reflected core simulation (fuel_t = 80 cm, 2 x 20 cm reflectors)
    model_refl = create_reflected_slab(
        fuel_thickness=fuel_t,
        reflector_thickness=refl_t,
        fuel_materials=fuel_mats,
        reflector_materials=refl_mats,
        symmetric=True,
    )
    mesh_refl = UniformMesh1D(geometry=model_refl.geometry, num_cells=240)
    refl_solver = HeterogeneousPowerIterationSolver(mesh=mesh_refl, material_model=model_refl)
    refl_result = refl_solver.solve()

    delta_k = refl_result.keff - bare_result.keff
    delta_k_pcm = delta_k * 1.0e5

    # Plotting comparison
    plot_bare_vs_reflector(
        mesh_bare=mesh_bare,
        bare_flux=bare_result.flux,
        mesh_refl=mesh_refl,
        refl_flux=refl_result.flux,
        fuel_thickness=fuel_t,
        reflector_thickness=refl_t,
        bare_keff=bare_result.keff,
        refl_keff=refl_result.keff,
        output_path=fig_dir / "bare_vs_reflector_keff.png",
    )

    comparison_data: dict[str, Any] = {
        "fuel_thickness_cm": fuel_t,
        "reflector_thickness_cm": refl_t,
        "bare_keff": bare_result.keff,
        "reflected_keff": refl_result.keff,
        "reflector_saving_delta_keff": delta_k,
        "reflector_saving_pcm": delta_k_pcm,
        "bare_peak_flux": float(np.max(bare_result.flux)),
        "reflected_peak_flux": float(np.max(refl_result.flux)),
    }

    with open(metric_dir / "reflector_savings.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    with open(table_dir / "reflector_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "bare_core", "reflected_core", "difference"])
        writer.writerow(["keff", f"{bare_result.keff:.8f}", f"{refl_result.keff:.8f}", f"{delta_k:.8f}"])
        writer.writerow(["pcm_reactivity_gain", "0.0", f"{delta_k_pcm:.1f}", f"{delta_k_pcm:.1f}"])

    return comparison_data


def study_critical_fuel_thickness(output_dir: str | Path = "results") -> dict[str, Any]:
    """Determine approximate numerical critical fuel core thickness with fixed reflector."""
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    metric_dir = out_path / "metrics"
    table_dir = out_path / "tables"

    for d in (fig_dir, metric_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    fuel_mats, refl_mats = get_default_heterogeneous_materials()
    refl_t = 20.0

    # Fuel thicknesses sweep bracketing critical thickness (~25 cm)
    fuel_thicknesses = [15.0, 20.0, 25.0, 30.0, 45.0, 60.0, 80.0]
    keff_values: list[float] = []

    for ft in fuel_thicknesses:
        model = create_reflected_slab(
            fuel_thickness=ft,
            reflector_thickness=refl_t,
            fuel_materials=fuel_mats,
            reflector_materials=refl_mats,
            symmetric=True,
        )
        # Choose cell count to maintain dx ~ 0.5 cm
        n_cells = int(round(model.geometry.length / 0.5))
        mesh = UniformMesh1D(geometry=model.geometry, num_cells=n_cells)
        res = HeterogeneousPowerIterationSolver(mesh=mesh, material_model=model).solve()
        keff_values.append(res.keff)

    # Numerical root finding for k_eff(T_crit) = 1.0 using monotonic spline interpolation
    cs = CubicSpline(fuel_thicknesses, np.array(keff_values) - 1.0)
    roots = cs.roots()
    valid_roots = [r for r in roots if min(fuel_thicknesses) <= r <= max(fuel_thicknesses)]
    crit_fuel_thickness = float(valid_roots[0]) if valid_roots else float(np.interp(1.0, keff_values, fuel_thicknesses))

    # Verify critical result with dedicated solve
    crit_model = create_reflected_slab(
        fuel_thickness=crit_fuel_thickness,
        reflector_thickness=refl_t,
        fuel_materials=fuel_mats,
        reflector_materials=refl_mats,
        symmetric=True,
    )
    crit_mesh = UniformMesh1D(
        geometry=crit_model.geometry,
        num_cells=int(round(crit_model.geometry.length / 0.5)),
    )
    crit_res = HeterogeneousPowerIterationSolver(mesh=crit_mesh, material_model=crit_model).solve()

    # Plot
    plot_critical_fuel_thickness(
        fuel_thicknesses=fuel_thicknesses,
        keff_values=keff_values,
        critical_thickness=crit_fuel_thickness,
        output_path=fig_dir / "critical_fuel_thickness.png",
    )

    data: dict[str, Any] = {
        "reflector_thickness_cm": refl_t,
        "fuel_thicknesses_sweep_cm": fuel_thicknesses,
        "keff_sweep": keff_values,
        "numerical_critical_fuel_thickness_cm": crit_fuel_thickness,
        "critical_keff_verification": crit_res.keff,
        "error_from_critical_pcm": abs(crit_res.keff - 1.0) * 1.0e5,
    }

    with open(metric_dir / "critical_thickness.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(table_dir / "critical_thickness_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fuel_thickness_cm", "keff"])
        for ft, k in zip(fuel_thicknesses, keff_values):
            writer.writerow([f"{ft:.2f}", f"{k:.8f}"])

    return data


def run_heterogeneous_parameter_study(output_dir: str | Path = "results") -> dict[str, Any]:
    """Execute parameter sensitivity sweeps for reflector thickness, reflector Sigma_a, fuel Sigma_a, and fuel nuSigma_f."""
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    metric_dir = out_path / "metrics"

    for d in (fig_dir, metric_dir):
        d.mkdir(parents=True, exist_ok=True)

    fuel_base, refl_base = get_default_heterogeneous_materials()
    fuel_t_base = 80.0
    refl_t_base = 20.0

    # 1. Sweep Reflector Thickness
    refl_t_vals = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    keff_refl_t = []
    for rt in refl_t_vals:
        m = create_reflected_slab(fuel_t_base, rt, fuel_base, refl_base)
        n = int(round(m.geometry.length / 0.5))
        keff_refl_t.append(HeterogeneousPowerIterationSolver(UniformMesh1D(m.geometry, n), m).solve().keff)

    # 2. Sweep Reflector Absorption
    refl_sa_vals = [0.0005, 0.001, 0.003, 0.006, 0.010, 0.020]
    keff_refl_sa = []
    for sa in refl_sa_vals:
        refl_m = MaterialProperties(D=refl_base.D, sigma_a=sa, nu_sigma_f=0.0, name="Reflector Var Sa")
        m = create_reflected_slab(fuel_t_base, refl_t_base, fuel_base, refl_m)
        keff_refl_sa.append(HeterogeneousPowerIterationSolver(UniformMesh1D(m.geometry, 240), m).solve().keff)

    # 3. Sweep Fuel Absorption
    fuel_sa_vals = [0.015, 0.018, 0.022, 0.026, 0.030]
    keff_fuel_sa = []
    for sa in fuel_sa_vals:
        fuel_m = MaterialProperties(D=fuel_base.D, sigma_a=sa, nu_sigma_f=fuel_base.nu_sigma_f, name="Fuel Var Sa")
        m = create_reflected_slab(fuel_t_base, refl_t_base, fuel_m, refl_base)
        keff_fuel_sa.append(HeterogeneousPowerIterationSolver(UniformMesh1D(m.geometry, 240), m).solve().keff)

    # 4. Sweep Fuel Production nuSigma_f
    fuel_nsf_vals = [0.022, 0.024, 0.026, 0.028, 0.030]
    keff_fuel_nsf = []
    for nsf in fuel_nsf_vals:
        fuel_m = MaterialProperties(D=fuel_base.D, sigma_a=fuel_base.sigma_a, nu_sigma_f=nsf, name="Fuel Var nsf")
        m = create_reflected_slab(fuel_t_base, refl_t_base, fuel_m, refl_base)
        keff_fuel_nsf.append(HeterogeneousPowerIterationSolver(UniformMesh1D(m.geometry, 240), m).solve().keff)

    results: dict[str, dict[str, list[float]]] = {
        "reflector_thickness": {"param_values": refl_t_vals, "keff_values": keff_refl_t},
        "reflector_sigma_a": {"param_values": refl_sa_vals, "keff_values": keff_refl_sa},
        "fuel_sigma_a": {"param_values": fuel_sa_vals, "keff_values": keff_fuel_sa},
        "fuel_nu_sigma_f": {"param_values": fuel_nsf_vals, "keff_values": keff_fuel_nsf},
    }

    plot_reflector_sensitivity(results, fig_dir / "reflector_sensitivity.png")

    with open(metric_dir / "heterogeneous_study.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def run_heterogeneous_convergence_study(
    mesh_cells_list: list[int] | None = None,
    output_dir: str | Path = "results",
) -> dict[str, Any]:
    """Execute spatial mesh convergence study on the heterogeneous core-reflector configuration."""
    cells_list = mesh_cells_list or [60, 120, 240, 480, 960]
    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fuel_mats, refl_mats = get_default_heterogeneous_materials()
    model = create_reflected_slab(80.0, 20.0, fuel_mats, refl_mats, symmetric=True)

    keffs = []
    dxs = []
    for n in cells_list:
        mesh = UniformMesh1D(model.geometry, n)
        res = HeterogeneousPowerIterationSolver(
            mesh=mesh,
            material_model=model,
            config=SolverConfig(tolerance_k=1.0e-11, tolerance_flux=1.0e-10),
        ).solve()
        keffs.append(res.keff)
        dxs.append(mesh.dx)

    # Richardson extrapolation reference from fine mesh
    k_ref = keffs[-1]
    errors = [abs(k - k_ref) for k in keffs[:-1]]
    dx_sub = dxs[:-1]

    plot_heterogeneous_convergence(dx_sub, errors, fig_dir / "heterogeneous_convergence.png")

    return {
        "cells": cells_list,
        "dx": dxs,
        "keff": keffs,
        "reference_keff": k_ref,
    }
