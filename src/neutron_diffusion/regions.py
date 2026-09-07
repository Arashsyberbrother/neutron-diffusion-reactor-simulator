"""Material region definitions and piecewise spatial mappings for heterogeneous 1D slabs."""

from dataclasses import dataclass
import numpy as np

from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D


@dataclass(frozen=True)
class MaterialRegion:
    """A contiguous spatial region endowed with homogeneous material cross sections.

    Parameters
    ----------
    name : str
        Descriptive label (e.g. 'fuel_core', 'reflector').
    x_min : float
        Left boundary of region in cm.
    x_max : float
        Right boundary of region in cm. Must satisfy $x_{\\min} < x_{\\max}$.
    materials : MaterialProperties
        Macroscopic cross sections and diffusion coefficient.

    Raises
    ------
    ValueError
        If $x_{\\min} \\ge x_{\\max}$ or negative coordinates are given.
    """

    name: str
    x_min: float
    x_max: float
    materials: MaterialProperties

    def __post_init__(self) -> None:
        if self.x_min < 0.0:
            raise ValueError(f"Region '{self.name}' left boundary cannot be negative, got {self.x_min}.")
        if self.x_max <= self.x_min:
            raise ValueError(
                f"Region '{self.name}' must have x_min < x_max, got [{self.x_min}, {self.x_max}]."
            )

    @property
    def thickness(self) -> float:
        """Region spatial thickness in cm."""
        return self.x_max - self.x_min

    @property
    def is_fissionable(self) -> bool:
        """True if region possesses non-zero neutron fission production."""
        return self.materials.nu_sigma_f > 0.0

    def contains(self, x: float) -> bool:
        """Check whether coordinate $x$ lies within $[x_{\\min}, x_{\\max}]$."""
        return self.x_min <= x <= self.x_max


@dataclass(frozen=True)
class PiecewiseMaterialModel:
    """Piecewise spatial representation of a multi-region heterogeneous reactor slab.

    Validates:
    1. Regions are strictly ordered and contiguous without spatial gaps.
    2. Regions do not overlap.
    3. Entire domain $[0, L]$ is spanned completely.

    Parameters
    ----------
    geometry : SlabGeometry
        Domain slab geometry.
    regions : tuple[MaterialRegion, ...]
        Ordered sequence of contiguous material regions.
    """

    geometry: SlabGeometry
    regions: tuple[MaterialRegion, ...]

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("PiecewiseMaterialModel must contain at least one MaterialRegion.")

        # Check domain start
        if abs(self.regions[0].x_min - self.geometry.x_min) > 1e-10:
            raise ValueError(
                f"First region must start at domain boundary {self.geometry.x_min}, "
                f"got {self.regions[0].x_min}."
            )

        # Check contiguous interfaces
        for i in range(len(self.regions) - 1):
            curr_r = self.regions[i]
            next_r = self.regions[i + 1]
            if abs(curr_r.x_max - next_r.x_min) > 1e-10:
                raise ValueError(
                    f"Discontinuity/gap or overlap detected between region '{curr_r.name}' "
                    f"(x_max={curr_r.x_max}) and region '{next_r.name}' (x_min={next_r.x_min})."
                )

        # Check domain end
        if abs(self.regions[-1].x_max - self.geometry.x_max) > 1e-10:
            raise ValueError(
                f"Last region must terminate at domain boundary {self.geometry.x_max}, "
                f"got {self.regions[-1].x_max}."
            )

    @property
    def num_regions(self) -> int:
        """Total number of material regions."""
        return len(self.regions)

    @property
    def interface_locations(self) -> list[float]:
        """List of interior material interface $x$-coordinates in cm."""
        return [r.x_max for r in self.regions[:-1]]

    def find_region_at(self, x: float) -> MaterialRegion:
        """Find the material region containing coordinate $x$.

        At exact interior interfaces $x_i$, the region to the left is chosen for $x_i$,
        unless $x=0$, in which case the first region is chosen.
        """
        if x < self.geometry.x_min - 1e-12 or x > self.geometry.x_max + 1e-12:
            raise ValueError(f"Coordinate {x} lies outside slab domain [0, {self.geometry.length}].")

        for r in self.regions:
            if r.x_min <= x <= r.x_max:
                return r
        # Fallback to last region if boundary precision edge
        return self.regions[-1]

    def sample_nodal_properties(
        self, mesh: UniformMesh1D
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample material properties onto a given spatial mesh.

        Parameters
        ----------
        mesh : UniformMesh1D
            Spatial grid.

        Returns
        -------
        D_nodal : np.ndarray
            Diffusion coefficient at all nodes (length $N+1$).
        sigma_a_nodal : np.ndarray
            Absorption cross section at all nodes (length $N+1$).
        nu_sigma_f_nodal : np.ndarray
            Fission production cross section at all nodes (length $N+1$).
        is_fissionable_nodal : np.ndarray
            Boolean mask identifying nodes residing in fissionable regions (length $N+1$).
        """
        n_pts = mesh.num_points
        dx = mesh.dx
        length = mesh.geometry.length
        sa_arr = np.zeros(n_pts, dtype=np.float64)
        nsf_arr = np.zeros(n_pts, dtype=np.float64)
        inv_d_arr = np.zeros(n_pts, dtype=np.float64)
        fiss_mask = np.zeros(n_pts, dtype=bool)

        for i in range(n_pts):
            # Control volume bounds around node x_i: [x_i - dx/2, x_i + dx/2]
            if i == 0:
                x_a, x_b = 0.0, 0.5 * dx
            elif i == n_pts - 1:
                x_a, x_b = length - 0.5 * dx, length
            else:
                x_a, x_b = mesh.x[i] - 0.5 * dx, mesh.x[i] + 0.5 * dx
            cv_len = x_b - x_a

            for r in self.regions:
                o_min = max(x_a, r.x_min)
                o_max = min(x_b, r.x_max)
                if o_max > o_min:
                    frac = (o_max - o_min) / cv_len
                    sa_arr[i] += frac * r.materials.sigma_a
                    nsf_arr[i] += frac * r.materials.nu_sigma_f
                    inv_d_arr[i] += frac / r.materials.D
                    if r.is_fissionable and frac > 1e-12:
                        fiss_mask[i] = True

        d_arr = 1.0 / inv_d_arr
        return d_arr, sa_arr, nsf_arr, fiss_mask


def create_reflected_slab(
    fuel_thickness: float,
    reflector_thickness: float,
    fuel_materials: MaterialProperties,
    reflector_materials: MaterialProperties,
    symmetric: bool = True,
) -> PiecewiseMaterialModel:
    """Create standard benchmark core-reflector configuration.

    Parameters
    ----------
    fuel_thickness : float
        Thickness of active fuel core in cm ($T_{\\text{fuel}} > 0$).
    reflector_thickness : float
        Thickness of non-multiplying reflector in cm ($T_{\\text{refl}} > 0$).
    fuel_materials : MaterialProperties
        Fissionable core material cross sections.
    reflector_materials : MaterialProperties
        Scattering reflector material cross sections (typically $\\nu\\Sigma_f = 0$).
    symmetric : bool, optional
        If True, builds symmetric core: [Reflector | Fuel Core | Reflector].
        If False, builds asymmetric core: [Fuel Core | Reflector].
        Default is True.

    Returns
    -------
    PiecewiseMaterialModel
        Validated piecewise multi-region model.
    """
    if fuel_thickness <= 0.0 or reflector_thickness <= 0.0:
        raise ValueError("Fuel and reflector thicknesses must be strictly positive.")

    if symmetric:
        total_length = 2.0 * reflector_thickness + fuel_thickness
        geom = SlabGeometry(length=total_length)
        r_left = MaterialRegion(
            name="left_reflector",
            x_min=0.0,
            x_max=reflector_thickness,
            materials=reflector_materials,
        )
        r_core = MaterialRegion(
            name="fuel_core",
            x_min=reflector_thickness,
            x_max=reflector_thickness + fuel_thickness,
            materials=fuel_materials,
        )
        r_right = MaterialRegion(
            name="right_reflector",
            x_min=reflector_thickness + fuel_thickness,
            x_max=total_length,
            materials=reflector_materials,
        )
        return PiecewiseMaterialModel(geometry=geom, regions=(r_left, r_core, r_right))
    else:
        total_length = fuel_thickness + reflector_thickness
        geom = SlabGeometry(length=total_length)
        r_core = MaterialRegion(
            name="fuel_core",
            x_min=0.0,
            x_max=fuel_thickness,
            materials=fuel_materials,
        )
        r_refl = MaterialRegion(
            name="reflector",
            x_min=fuel_thickness,
            x_max=total_length,
            materials=reflector_materials,
        )
        return PiecewiseMaterialModel(geometry=geom, regions=(r_core, r_refl))
