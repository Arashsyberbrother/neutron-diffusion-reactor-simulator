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
    """Verify that the numerical critical fuel thickness workflow finds an active core size where k_eff ~ 1.0."""
    data = study_critical_fuel_thickness(output_dir=tmp_path)

    crit_t = data["numerical_critical_fuel_thickness_cm"]
    crit_k = data["critical_keff_verification"]

    assert 15.0 < crit_t < 40.0
    # Achieves approximate numerical criticality within 500 pcm
    assert pytest.approx(1.0, abs=0.005) == crit_k


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
