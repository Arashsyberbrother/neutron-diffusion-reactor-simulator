"""Unit tests for MaterialRegion and PiecewiseMaterialModel abstractions."""

import pytest
import numpy as np

from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.regions import (
    MaterialRegion,
    PiecewiseMaterialModel,
    create_reflected_slab,
)


@pytest.fixture
def test_materials() -> tuple[MaterialProperties, MaterialProperties]:
    fuel = MaterialProperties(D=1.2, sigma_a=0.02, nu_sigma_f=0.025, name="Fuel")
    refl = MaterialProperties(D=0.8, sigma_a=0.005, nu_sigma_f=0.0, name="Reflector")
    return fuel, refl


def test_material_region_properties(test_materials) -> None:
    fuel, refl = test_materials
    r1 = MaterialRegion(name="core", x_min=10.0, x_max=50.0, materials=fuel)
    assert r1.thickness == 40.0
    assert r1.is_fissionable is True
    assert r1.contains(10.0) is True
    assert r1.contains(30.0) is True
    assert r1.contains(50.0) is True
    assert r1.contains(5.0) is False

    r2 = MaterialRegion(name="refl", x_min=0.0, x_max=10.0, materials=refl)
    assert r2.is_fissionable is False


def test_material_region_validation(test_materials) -> None:
    fuel, _ = test_materials
    with pytest.raises(ValueError, match="left boundary cannot be negative"):
        MaterialRegion(name="bad", x_min=-5.0, x_max=10.0, materials=fuel)

    with pytest.raises(ValueError, match="must have x_min < x_max"):
        MaterialRegion(name="bad", x_min=10.0, x_max=5.0, materials=fuel)


def test_piecewise_material_model_valid(test_materials) -> None:
    fuel, refl = test_materials
    geom = SlabGeometry(length=100.0)
    r_left = MaterialRegion(name="refl_left", x_min=0.0, x_max=20.0, materials=refl)
    r_core = MaterialRegion(name="fuel_core", x_min=20.0, x_max=80.0, materials=fuel)
    r_right = MaterialRegion(name="refl_right", x_min=80.0, x_max=100.0, materials=refl)

    model = PiecewiseMaterialModel(geometry=geom, regions=(r_left, r_core, r_right))
    assert model.num_regions == 3
    assert model.interface_locations == [20.0, 80.0]

    # Query coordinate inside region
    assert model.find_region_at(10.0).name == "refl_left"
    assert model.find_region_at(50.0).name == "fuel_core"
    assert model.find_region_at(90.0).name == "refl_right"


def test_piecewise_material_model_gaps_and_overlaps(test_materials) -> None:
    fuel, refl = test_materials
    geom = SlabGeometry(length=100.0)

    # Gap between 20 and 25
    r1 = MaterialRegion(name="r1", x_min=0.0, x_max=20.0, materials=refl)
    r2_gap = MaterialRegion(name="r2", x_min=25.0, x_max=100.0, materials=fuel)
    with pytest.raises(ValueError, match="Discontinuity/gap or overlap"):
        PiecewiseMaterialModel(geometry=geom, regions=(r1, r2_gap))

    # Overlap between 15 and 20
    r2_overlap = MaterialRegion(name="r2", x_min=15.0, x_max=100.0, materials=fuel)
    with pytest.raises(ValueError, match="Discontinuity/gap or overlap"):
        PiecewiseMaterialModel(geometry=geom, regions=(r1, r2_overlap))


def test_create_reflected_slab_helper(test_materials) -> None:
    fuel, refl = test_materials
    sym_model = create_reflected_slab(60.0, 20.0, fuel, refl, symmetric=True)
    assert sym_model.geometry.length == 100.0
    assert sym_model.num_regions == 3
    assert sym_model.interface_locations == [20.0, 80.0]

    asym_model = create_reflected_slab(60.0, 20.0, fuel, refl, symmetric=False)
    assert asym_model.geometry.length == 80.0
    assert asym_model.num_regions == 2
    assert asym_model.interface_locations == [60.0]


def test_nodal_property_sampling(test_materials) -> None:
    fuel, refl = test_materials
    model = create_reflected_slab(60.0, 20.0, fuel, refl, symmetric=True)
    mesh = UniformMesh1D(model.geometry, num_cells=10)

    d_arr, sa_arr, nsf_arr, fiss_mask = model.sample_nodal_properties(mesh)
    assert len(d_arr) == 11
    assert len(sa_arr) == 11
    assert len(nsf_arr) == 11
    assert len(fiss_mask) == 11

    # Nodal D in reflector should equal reflector D
    assert d_arr[0] == pytest.approx(refl.D)
    assert d_arr[1] == pytest.approx(refl.D)
    # Nodal D in fuel core center should equal fuel D
    assert d_arr[5] == pytest.approx(fuel.D)
    assert nsf_arr[0] == 0.0
    assert nsf_arr[5] == pytest.approx(fuel.nu_sigma_f)
    assert bool(fiss_mask[0]) is False
    assert bool(fiss_mask[5]) is True
