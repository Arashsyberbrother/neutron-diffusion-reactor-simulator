"""Tests for Version 2 command-line interface commands."""

from pathlib import Path
from neutron_diffusion.cli import main


def test_cli_heterogeneous_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_het"
    exit_code = main(["heterogeneous", "--output-dir", str(out_dir)])
    assert exit_code == 0
    assert (out_dir / "figures" / "core_reflector_flux.png").exists()
    assert (out_dir / "metrics" / "heterogeneous_summary.json").exists()


def test_cli_reflector_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_refl"
    exit_code = main(["reflector", "--output-dir", str(out_dir)])
    assert exit_code == 0
    assert (out_dir / "figures" / "bare_vs_reflector_keff.png").exists()
    assert (out_dir / "metrics" / "reflector_savings.json").exists()


def test_cli_critical_thickness_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_crit"
    exit_code = main(["critical-thickness", "--output-dir", str(out_dir)])
    assert exit_code == 0
    assert (out_dir / "figures" / "critical_fuel_thickness.png").exists()
    assert (out_dir / "metrics" / "critical_thickness.json").exists()
