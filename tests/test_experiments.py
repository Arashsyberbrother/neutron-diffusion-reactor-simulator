"""Tests for scientific experiments and reactor physics sensitivity studies."""

import json
from pathlib import Path
import pytest

from neutron_diffusion.config import SimulationConfig
from neutron_diffusion.experiments import (
    run_baseline_experiment,
    run_mesh_convergence_study,
    run_parameter_study,
)


def test_run_baseline_experiment_artifacts(tmp_path: Path) -> None:
    test_out = tmp_path / "results_test"
    cfg = SimulationConfig(num_cells=40, max_iterations=500)

    summary = run_baseline_experiment(config=cfg, output_dir=test_out)

    assert summary["convergence"]["converged"]
    assert summary["results"]["numerical_keff"] > 1.0

    # Verify directory structure and generated files
    assert (test_out / "figures" / "flux_comparison.png").exists()
    assert (test_out / "figures" / "power_distribution.png").exists()
    assert (test_out / "figures" / "power_iteration_convergence.png").exists()
    assert (test_out / "metrics" / "baseline_summary.json").exists()
    assert (test_out / "tables" / "baseline_flux.csv").exists()

    # Verify JSON content
    with open(test_out / "metrics" / "baseline_summary.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["parameters"]["length_cm"] == 100.0


def test_parameter_sensitivity_trends(tmp_path: Path) -> None:
    """Verify fundamental reactor physics principles from parameter sweeps:
    - Increasing L reduces leakage -> k_eff increases
    - Increasing Sigma_a increases absorption -> k_eff decreases
    - Increasing D increases leakage -> k_eff decreases
    - Increasing nuSigma_f increases production -> k_eff increases
    """
    test_out = tmp_path / "sens_test"
    results = run_parameter_study(output_dir=test_out)

    # 1. Slab length L (monotonic increase)
    l_keff = results["length"]["keff_values"]
    for i in range(len(l_keff) - 1):
        assert l_keff[i + 1] > l_keff[i], "k_eff must increase with core size L due to lower leakage"

    # 2. Absorption Sigma_a (monotonic decrease)
    sa_keff = results["sigma_a"]["keff_values"]
    for i in range(len(sa_keff) - 1):
        assert sa_keff[i + 1] < sa_keff[i], "k_eff must decrease with increased absorption cross section"

    # 3. Diffusion coefficient D (monotonic decrease)
    d_keff = results["D"]["keff_values"]
    for i in range(len(d_keff) - 1):
        assert d_keff[i + 1] < d_keff[i], "k_eff must decrease with higher diffusion coefficient (higher leakage)"

    # 4. Fission production nuSigma_f (monotonic increase)
    nf_keff = results["nu_sigma_f"]["keff_values"]
    for i in range(len(nf_keff) - 1):
        assert nf_keff[i + 1] > nf_keff[i], "k_eff must increase with higher fission production cross section"
