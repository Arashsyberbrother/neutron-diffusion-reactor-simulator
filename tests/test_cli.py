"""Tests for command-line interface entry points."""

from pathlib import Path
import pytest

from neutron_diffusion.cli import main


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "usage:" in captured.err or "neutron-diffusion" in captured.out


def test_cli_baseline_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_results"
    exit_code = main(["baseline", "--cells", "30", "--output-dir", str(out_dir)])
    assert exit_code == 0
    assert (out_dir / "metrics" / "baseline_summary.json").exists()


def test_cli_verify_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_verify"
    exit_code = main(["verify", "--output-dir", str(out_dir)])
    assert exit_code == 0
    assert (out_dir / "figures" / "flux_comparison.png").exists()


def test_cli_convergence_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_conv"
    exit_code = main(["convergence", "--cells", "20", "40", "--output-dir", str(out_dir)])
    assert exit_code == 0
    assert (out_dir / "metrics" / "mesh_convergence.json").exists()
