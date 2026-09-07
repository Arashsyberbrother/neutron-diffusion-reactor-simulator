"""Tests for spatial mesh discretization and integration utilities."""

import math
import pytest
import numpy as np

from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.mesh import UniformMesh1D


def test_mesh_spacing_and_dimensions() -> None:
    geom = SlabGeometry(length=120.0)
    mesh = UniformMesh1D(geometry=geom, num_cells=60)

    assert mesh.dx == pytest.approx(2.0, rel=1e-12)
    assert mesh.num_points == 61
    assert mesh.num_interior == 59
    assert len(mesh.x) == 61
    assert len(mesh.x_interior) == 59
    assert mesh.x[0] == 0.0
    assert mesh.x[-1] == 120.0
    assert mesh.x_interior[0] == pytest.approx(2.0, rel=1e-12)
    assert mesh.x_interior[-1] == pytest.approx(118.0, rel=1e-12)


def test_mesh_validation() -> None:
    geom = SlabGeometry(length=50.0)
    with pytest.raises(ValueError, match="Number of mesh cells must be at least 2"):
        UniformMesh1D(geometry=geom, num_cells=1)


def test_mesh_composite_trapezoid_integration() -> None:
    # Test integration of sin(pi*x/L) over [0, L]
    # Exact integral: \int_0^L sin(pi*x/L) dx = 2*L / pi
    L = 100.0
    geom = SlabGeometry(length=L)
    mesh = UniformMesh1D(geometry=geom, num_cells=200)

    f_full = np.sin(math.pi * mesh.x / L)
    exact_integral = 2.0 * L / math.pi
    num_integral = mesh.integrate(f_full)

    # Trapezoid rule on 200 intervals has O(dx^2) error ~ 1e-4
    assert pytest.approx(exact_integral, rel=1e-4) == num_integral

    # Test integration using interior values only
    f_int = f_full[1:-1]
    int_from_interior = mesh.integrate(f_int)
    assert pytest.approx(num_integral, rel=1e-12) == int_from_interior


def test_mesh_reconstruct_full_field() -> None:
    geom = SlabGeometry(length=10.0)
    mesh = UniformMesh1D(geometry=geom, num_cells=5)

    interior = np.array([1.0, 2.0, 3.0, 4.0])
    full = mesh.reconstruct_full_field(interior, left_val=-1.0, right_val=9.0)

    assert len(full) == 6
    assert full[0] == -1.0
    assert full[-1] == 9.0
    assert np.all(full[1:-1] == interior)

    with pytest.raises(ValueError, match="Expected interior field of length"):
        mesh.reconstruct_full_field(np.array([1.0, 2.0]))
