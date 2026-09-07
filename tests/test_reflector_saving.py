"""Tests for reflector saving effect and numerical critical core thickness determination."""

import pytest
from pathlib import Path

from neutron_diffusion.reflector_study import (
    compare_bare_vs_reflected,
    get_default_heterogeneous_materials,
    study_critical_fuel_thickness,
)
from neutron_diffusion.regions import create_reflected_slab
from neutron_diffusion.heterogeneous_solver import HeterogeneousPowerIterationSolver
from neutron_diffusion.mesh import UniformMesh1D


def test_reflector_saving_reactivity_gain(tmp_path: Path) -> None:
    """Verify that adding a scattering reflector to a bare fuel slab strictly increases k_eff (reflector savings)."""
    comp = compare_bare_vs_reflected(output_dir=tmp_path)

    bare_k = comp["bare_keff"]
    refl_k = comp["reflected_keff"]
    delta_k = comp["reflector_saving_delta_keff"]

    # Reflector returns leakage neutrons back into the core, increasing k_eff
    assert refl_k > bare_k
    assert delta_k > 0.0
    assert comp["reflector_saving_pcm"] > 500.0  # Significant reactivity gain (> 500 pcm)


def test_critical_fuel_thickness_solver(tmp_path: Path) -> None:
    """Verify that the numerical critical fuel thickness workflow robustly brackets and converges to k_eff ~ 1.0."""
    import numpy as np

    data = study_critical_fuel_thickness(output_dir=tmp_path)

    crit_t = data["numerical_critical_fuel_thickness_cm"]
    crit_k = data["critical_keff_verification"]
    bracket = data["initial_bracket_cm"]
    bracket_k = data["bracket_keff"]
    keff_sweep = np.array(data["keff_sweep"])

    # 1. Valid bracket exists
    assert bracket[0] < bracket[1], "Bracket lower bound must be strictly less than upper bound"
    assert bracket_k[0] < 1.0, f"Bracket lower bound must be subcritical, got {bracket_k[0]}"
    assert bracket_k[1] > 1.0, f"Bracket upper bound must be supercritical, got {bracket_k[1]}"

    # 2. Monotonic increase over parameter sweep
    diffs = np.diff(keff_sweep)
    assert np.all(diffs > 0.0), f"k_eff must increase strictly monotonically with fuel thickness, got diffs {diffs}"

    # 3. Returned root lies inside the initial bracket
    assert bracket[0] <= crit_t <= bracket[1], f"Root {crit_t} must lie within bracket {bracket}"

    # 4. Direct k_eff satisfies accuracy target |k_eff - 1| <= 1e-4
    err = abs(crit_k - 1.0)
    assert err <= 1.0e-4, f"Direct k_eff error {err:.2e} exceeds target tolerance 1e-4"

    # 5. Local neighborhood verification around T_crit
    local = data["local_verification"]
    assert local["keff_minus_0_5"] < 1.0, "k_eff at T_crit - 0.5 cm must be strictly subcritical"
    assert abs(local["keff_crit"] - 1.0) <= 1.0e-4, "k_eff at T_crit must be within 1e-4 of critical"
    assert local["keff_plus_0_5"] > 1.0, "k_eff at T_crit + 0.5 cm must be strictly supercritical"


def test_critical_fuel_thickness_reproducibility(tmp_path: Path) -> None:
    """Verify that the numerical bisection root calculation is deterministic and reproducible."""
    d1 = study_critical_fuel_thickness(output_dir=tmp_path / "run1")
    d2 = study_critical_fuel_thickness(output_dir=tmp_path / "run2")

    t1 = d1["numerical_critical_fuel_thickness_cm"]
    t2 = d2["numerical_critical_fuel_thickness_cm"]
    k1 = d1["critical_keff_verification"]
    k2 = d2["critical_keff_verification"]

    assert abs(t1 - t2) < 1.0e-7, f"T_crit must be reproducible within machine resolution: {t1} vs {t2}"
    assert abs(k1 - k2) < 1.0e-7, f"k_eff must be reproducible: {k1} vs {k2}"



def test_reflector_thickness_monotonicity() -> None:
    """Verify that increasing reflector thickness (with low absorption) monotonically increases reactivity."""
    fuel, refl = get_default_heterogeneous_materials()
    fuel_t = 60.0

    keffs = []
    refl_thicknesses = [5.0, 15.0, 30.0]
    for rt in refl_thicknesses:
        model = create_reflected_slab(fuel_t, rt, fuel, refl, symmetric=True)
        mesh = UniformMesh1D(model.geometry, num_cells=int(round(model.geometry.length / 0.5)))
        res = HeterogeneousPowerIterationSolver(mesh=mesh, material_model=model).solve()
        keffs.append(res.keff)

    assert keffs[1] > keffs[0], "Increasing reflector from 5 to 15 cm must increase k_eff"
    assert keffs[2] > keffs[1], "Increasing reflector from 15 to 30 cm must increase k_eff"
