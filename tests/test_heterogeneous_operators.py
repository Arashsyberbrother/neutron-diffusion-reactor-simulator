"""Tests for heterogeneous finite difference operators and conservative interface current."""

import pytest
import numpy as np

from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.operators import LossOperator
from neutron_diffusion.regions import (
    MaterialRegion,
    PiecewiseMaterialModel,
    create_reflected_slab,
)
from neutron_diffusion.heterogeneous_operators import (
    HeterogeneousFissionOperator,
    HeterogeneousLossOperator,
    harmonic_mean,
)


def test_harmonic_mean() -> None:
    # If both values are equal, harmonic mean is equal
    assert pytest.approx(1.5) == harmonic_mean(1.5, 1.5)

    # D1 = 1.0, D2 = 2.0 -> 2*(1*2)/(1+2) = 4/3 = 1.3333...
    assert pytest.approx(4.0 / 3.0, rel=1e-12) == harmonic_mean(1.0, 2.0)

    with pytest.raises(ValueError, match="must be positive"):
        harmonic_mean(-1.0, 2.0)


def test_heterogeneous_operator_homogeneous_equivalence() -> None:
    """CRITICAL VERIFICATION: When material properties across regions are identical,
    the heterogeneous operator must match the homogeneous baseline operator to machine precision."""
    L = 100.0
    geom = SlabGeometry(length=L)
    mats = MaterialProperties(D=1.2, sigma_a=0.022, nu_sigma_f=0.026)
    mesh = UniformMesh1D(geometry=geom, num_cells=40)

    # 1. Homogeneous baseline operator
    hom_op = LossOperator(mesh=mesh, materials=mats)
    matrix_hom = hom_op.to_dense()

    # 2. Heterogeneous operator with 3 identical material regions
    r1 = MaterialRegion(name="r1", x_min=0.0, x_max=30.0, materials=mats)
    r2 = MaterialRegion(name="r2", x_min=30.0, x_max=70.0, materials=mats)
    r3 = MaterialRegion(name="r3", x_min=70.0, x_max=100.0, materials=mats)
    model = PiecewiseMaterialModel(geometry=geom, regions=(r1, r2, r3))

    het_op = HeterogeneousLossOperator(mesh=mesh, material_model=model)
    matrix_het = het_op.to_dense()

    # Must be mathematically identical
    assert np.allclose(matrix_hom, matrix_het, rtol=1e-14, atol=1e-14)


def test_heterogeneous_matrix_properties() -> None:
    fuel = MaterialProperties(D=1.25, sigma_a=0.025, nu_sigma_f=0.028)
    refl = MaterialProperties(D=0.80, sigma_a=0.004, nu_sigma_f=0.0)
    model = create_reflected_slab(60.0, 20.0, fuel, refl, symmetric=True)
    mesh = UniformMesh1D(model.geometry, num_cells=50)

    op = HeterogeneousLossOperator(mesh=mesh, material_model=model)
    matrix = op.to_dense()
    n = mesh.num_interior
    assert matrix.shape == (n, n)

    # 1. Matrix Symmetry: A = A^T
    assert np.allclose(matrix, matrix.T, atol=1e-14)

    # 2. Strict diagonal dominance: |A_ii| > sum_{j != i} |A_ij|
    row_sums = np.sum(np.abs(matrix), axis=1)
    diag_vals = np.abs(np.diag(matrix))
    off_diag_sums = row_sums - diag_vals
    assert np.all(diag_vals > off_diag_sums)

    # 3. Positive definiteness: all eigenvalues > 0
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert np.all(eigenvalues > 0.0)


def test_heterogeneous_face_currents() -> None:
    fuel = MaterialProperties(D=1.0, sigma_a=0.02, nu_sigma_f=0.025)
    refl = MaterialProperties(D=0.5, sigma_a=0.005, nu_sigma_f=0.0)
    model = create_reflected_slab(60.0, 20.0, fuel, refl, symmetric=True)
    mesh = UniformMesh1D(model.geometry, num_cells=20)

    op = HeterogeneousLossOperator(mesh=mesh, material_model=model)

    # Monotonically increasing linear test flux
    flux = np.linspace(0.0, 1.0, mesh.num_interior)
    currents = op.compute_face_currents(flux)

    # J = -D * dphi/dx
    # For increasing flux (dphi/dx > 0) in interior, J in positive x-direction must be negative
    # At the right boundary, full_flux drops to 0.0, so J > 0 (outward leakage)
    assert len(currents) == mesh.num_cells
    assert np.all(currents[:-1] <= 0.0)
    assert currents[-1] > 0.0


def test_heterogeneous_fission_operator() -> None:
    fuel = MaterialProperties(D=1.0, sigma_a=0.02, nu_sigma_f=0.025)
    refl = MaterialProperties(D=0.5, sigma_a=0.005, nu_sigma_f=0.0)
    model = create_reflected_slab(60.0, 20.0, fuel, refl, symmetric=True)
    mesh = UniformMesh1D(model.geometry, num_cells=50)

    fop = HeterogeneousFissionOperator(mesh=mesh, material_model=model)
    ones_flux = np.ones(mesh.num_interior)
    source = fop.apply(ones_flux)

    # Source must be zero in interior reflector nodes and equal nu_sigma_f in core nodes
    for xi, s in zip(mesh.x_interior, source):
        if xi < 20.0 - 1e-5 or xi > 80.0 + 1e-5:
            assert s == 0.0
        elif 20.0 + 1e-5 < xi < 80.0 - 1e-5:
            assert pytest.approx(fuel.nu_sigma_f) == s
        else:
            # Interface node between fuel and reflector
            assert pytest.approx(0.5 * fuel.nu_sigma_f) == s
