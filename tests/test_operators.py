"""Tests for discrete differential loss operator and tridiagonal solver."""

import pytest
import numpy as np
import scipy.linalg

from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.operators import (
    FissionOperator,
    LossOperator,
    solve_tridiagonal_thomas,
)


@pytest.fixture
def baseline_setup() -> tuple[UniformMesh1D, MaterialProperties]:
    geom = SlabGeometry(length=100.0)
    mesh = UniformMesh1D(geometry=geom, num_cells=50)
    mats = MaterialProperties(D=1.0, sigma_a=0.02, nu_sigma_f=0.025)
    return mesh, mats


def test_loss_operator_matrix_properties(baseline_setup) -> None:
    mesh, mats = baseline_setup
    op = LossOperator(mesh=mesh, materials=mats)

    matrix = op.to_dense()
    n = mesh.num_interior
    assert matrix.shape == (n, n)

    # 1. Symmetry
    assert np.allclose(matrix, matrix.T, atol=1e-14)

    # 2. Diagonal and off-diagonal coefficients
    dx = mesh.dx
    expected_diag = 2.0 * mats.D / (dx * dx) + mats.sigma_a
    expected_offdiag = -mats.D / (dx * dx)

    assert pytest.approx(expected_diag, rel=1e-12) == op.diagonal_value
    assert pytest.approx(expected_offdiag, rel=1e-12) == op.off_diagonal_value

    for i in range(n):
        assert pytest.approx(expected_diag, rel=1e-12) == matrix[i, i]
        if i > 0:
            assert pytest.approx(expected_offdiag, rel=1e-12) == matrix[i, i - 1]
        if i < n - 1:
            assert pytest.approx(expected_offdiag, rel=1e-12) == matrix[i, i + 1]

    # 3. Strict diagonal dominance: |A_ii| > sum_{j != i} |A_ij| (since Sigma_a > 0)
    row_sums = np.sum(np.abs(matrix), axis=1)
    diag_vals = np.abs(np.diag(matrix))
    off_diag_sums = row_sums - diag_vals
    assert np.all(diag_vals > off_diag_sums)

    # 4. Positive definiteness: all eigenvalues must be strictly positive
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert np.all(eigenvalues > 0.0)


def test_thomas_algorithm_vs_scipy_solve(baseline_setup) -> None:
    mesh, mats = baseline_setup
    op = LossOperator(mesh=mesh, materials=mats)
    dense_matrix = op.to_dense()

    # Generate test RHS vector
    rng = np.random.default_rng(42)
    rhs = rng.uniform(0.5, 2.0, size=mesh.num_interior)

    # Solve using custom O(N) Thomas algorithm
    sol_thomas = op.solve(rhs)

    # Solve using SciPy's standard linear solver
    sol_scipy = scipy.linalg.solve(dense_matrix, rhs)

    # Must agree to near machine precision
    assert np.allclose(sol_thomas, sol_scipy, rtol=1e-12, atol=1e-13)


def test_operator_apply_consistency(baseline_setup) -> None:
    mesh, mats = baseline_setup
    op = LossOperator(mesh=mesh, materials=mats)
    matrix = op.to_dense()

    rng = np.random.default_rng(123)
    vec = rng.standard_normal(mesh.num_interior)

    result_apply = op.apply(vec)
    result_matmul = matrix @ vec

    assert np.allclose(result_apply, result_matmul, atol=1e-14)


def test_thomas_zero_pivot_exception() -> None:
    # Test that Thomas solver raises ZeroDivisionError on zero pivot
    lower = np.array([1.0])
    diag = np.array([0.0, 1.0])  # Zero pivot at index 0
    upper = np.array([1.0])
    rhs = np.array([1.0, 1.0])

    with pytest.raises(ZeroDivisionError):
        solve_tridiagonal_thomas(lower, diag, upper, rhs)


def test_fission_operator(baseline_setup) -> None:
    mesh, mats = baseline_setup
    fop = FissionOperator(mesh=mesh, materials=mats)

    vec = np.ones(mesh.num_interior)
    source = fop.apply(vec)
    assert np.all(source == mats.nu_sigma_f)

    # Total production should equal nu_sigma_f * integral of flux
    tot_prod = fop.total_production(vec)
    tot_flux_int = mesh.integrate(vec)
    assert pytest.approx(mats.nu_sigma_f * tot_flux_int, rel=1e-12) == tot_prod
