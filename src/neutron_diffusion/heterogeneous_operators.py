"""Conservative finite-volume-style differential operators for heterogeneous 1D diffusion."""

from dataclasses import dataclass
import numpy as np

from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.operators import solve_tridiagonal_thomas
from neutron_diffusion.regions import PiecewiseMaterialModel


def harmonic_mean(d_left: float, d_right: float) -> float:
    """Compute harmonic average of diffusion coefficients across cell interface.

    $$D_{i+1/2} = \\frac{2 D_i D_{i+1}}{D_i + D_{i+1}}$$

    Parameters
    ----------
    d_left : float
        Diffusion coefficient on left side ($D > 0$).
    d_right : float
        Diffusion coefficient on right side ($D > 0$).

    Returns
    -------
    float
        Harmonic mean interface diffusion coefficient.
    """
    if d_left <= 0.0 or d_right <= 0.0:
        raise ValueError(f"Diffusion coefficients must be positive, got ({d_left}, {d_right}).")
    return (2.0 * d_left * d_right) / (d_left + d_right)


@dataclass(frozen=True)
class HeterogeneousLossOperator:
    """Conservative discretization of heterogeneous loss operator:
    $-\\frac{d}{dx}\\left(D(x) \\frac{d\\phi}{dx}\\right) + \\Sigma_a(x) \\phi$.

    Uses harmonic interface diffusion coefficients at cell faces $x_{i+1/2}$:
    $$D_{i+1/2} = \\frac{2 D_i D_{i+1}}{D_i + D_{i+1}}$$

    The net neutron current in the positive $x$-direction at face $i+1/2$ is:
    $$J_{i+1/2} = -D_{i+1/2} \\frac{\\phi_{i+1} - \\phi_i}{\\Delta x}$$

    The resulting interior system matrix $\\mathbf{A} \\in \\mathbb{R}^{(N-1)\\times(N-1)}$ is:
    - Main diagonal: $A_{i,i} = \\frac{D_{i-1/2} + D_{i+1/2}}{\\Delta x^2} + \\Sigma_{a,i}$
    - Sub-diagonal: $A_{i, i-1} = -\\frac{D_{i-1/2}}{\\Delta x^2}$
    - Super-diagonal: $A_{i, i+1} = -\\frac{D_{i+1/2}}{\\Delta x^2}$

    Parameters
    ----------
    mesh : UniformMesh1D
        Spatial grid.
    material_model : PiecewiseMaterialModel
        Piecewise material domain model.
    """

    mesh: UniformMesh1D
    material_model: PiecewiseMaterialModel

    @property
    def num_interior(self) -> int:
        """Number of interior grid unknowns $n = N - 1$."""
        return self.mesh.num_interior

    def compute_face_diffusion_coefficients(self) -> np.ndarray:
        """Compute harmonic mean face diffusion coefficients $D_{i+1/2}$ for $i = 0, \\dots, N-1$.

        Returns
        -------
        np.ndarray
            Array of length $N$ containing face diffusion coefficients at $x_{1/2}, x_{3/2}, \\dots, x_{N-1/2}$.
        """
        d_nodal, _, _, _ = self.material_model.sample_nodal_properties(self.mesh)
        # N intervals -> N faces
        d_faces = np.empty(self.mesh.num_cells, dtype=np.float64)
        for i in range(self.mesh.num_cells):
            d_faces[i] = harmonic_mean(d_nodal[i], d_nodal[i + 1])
        return d_faces

    @property
    def bands(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tridiagonal bands (lower, main_diag, upper) for interior nodes $i = 1, \\dots, N-1$."""
        n = self.num_interior
        dx = self.mesh.dx
        dx_sq = dx * dx

        d_faces = self.compute_face_diffusion_coefficients()
        # d_faces[i] corresponds to face between node i and node i+1 (i.e. x_{i+1/2})
        # For interior node i (1-indexed, i = 1, ..., N-1):
        # Left face is i - 1/2 -> d_faces[i - 1]
        # Right face is i + 1/2 -> d_faces[i]

        _, sa_nodal, _, _ = self.material_model.sample_nodal_properties(self.mesh)
        sa_interior = sa_nodal[1:-1]

        # Diagonals
        diag = np.empty(n, dtype=np.float64)
        for i in range(n):
            node_idx = i + 1  # 1-indexed node
            d_left_face = d_faces[node_idx - 1]
            d_right_face = d_faces[node_idx]
            diag[i] = (d_left_face + d_right_face) / dx_sq + sa_interior[i]

        # Sub-diagonal: A[i, i-1] = -d_left_face / dx_sq (length n-1)
        lower = np.empty(n - 1, dtype=np.float64)
        for i in range(n - 1):
            node_idx = i + 2  # represents row i+1 of interior matrix
            d_left_face = d_faces[node_idx - 1]
            lower[i] = -d_left_face / dx_sq

        # Super-diagonal: A[i, i+1] = -d_right_face / dx_sq (length n-1)
        upper = np.empty(n - 1, dtype=np.float64)
        for i in range(n - 1):
            node_idx = i + 1  # represents row i of interior matrix
            d_right_face = d_faces[node_idx]
            upper[i] = -d_right_face / dx_sq

        return lower, diag, upper

    def apply(self, interior_flux: np.ndarray) -> np.ndarray:
        """Matrix-vector multiplication $\\mathbf{A} \\boldsymbol{\\phi}$."""
        lower, diag, upper = self.bands
        n = self.num_interior
        result = np.empty(n, dtype=np.float64)
        result[0] = diag[0] * interior_flux[0] + upper[0] * interior_flux[1]
        result[1:-1] = (
            lower[:-1] * interior_flux[:-2]
            + diag[1:-1] * interior_flux[1:-1]
            + upper[1:] * interior_flux[2:]
        )
        result[-1] = lower[-1] * interior_flux[-2] + diag[-1] * interior_flux[-1]
        return result

    def solve(self, source: np.ndarray) -> np.ndarray:
        """Solve fixed-source diffusion equation $\\mathbf{A} \\boldsymbol{\\phi} = \\mathbf{s}$ via Thomas algorithm."""
        lower, diag, upper = self.bands
        return solve_tridiagonal_thomas(lower, diag, upper, source)

    def to_dense(self) -> np.ndarray:
        """Construct dense $(N-1)\\times(N-1)$ matrix for inspection and validation."""
        n = self.num_interior
        lower, diag, upper = self.bands
        matrix = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(matrix, diag)
        np.fill_diagonal(matrix[:-1, 1:], upper)
        np.fill_diagonal(matrix[1:, :-1], lower)
        return matrix

    def compute_face_currents(self, interior_flux: np.ndarray) -> np.ndarray:
        """Compute net neutron currents $J_{i+1/2} = -D_{i+1/2} \\frac{\\phi_{i+1} - \\phi_i}{\\Delta x}$ across all faces.

        Both derivatives and currents are defined with the standard positive $x$-direction convention.

        Parameters
        ----------
        interior_flux : np.ndarray
            Interior flux vector of length $N-1$.

        Returns
        -------
        np.ndarray
            Face currents of length $N$ at $x_{1/2}, x_{3/2}, \\dots, x_{N-1/2}$.
        """
        full_flux = self.mesh.reconstruct_full_field(interior_flux, 0.0, 0.0)
        d_faces = self.compute_face_diffusion_coefficients()
        dx = self.mesh.dx

        # J = -D * dphi/dx
        currents = np.empty(self.mesh.num_cells, dtype=np.float64)
        for i in range(self.mesh.num_cells):
            dphi_dx = (full_flux[i + 1] - full_flux[i]) / dx
            currents[i] = -d_faces[i] * dphi_dx
        return currents


@dataclass(frozen=True)
class HeterogeneousFissionOperator:
    """Neutron fission production operator $\\mathcal{F}\\phi = \\nu\\Sigma_f(x) \\phi(x)$ for heterogeneous media.

    In non-multiplying zones (e.g. reflector regions), $\\nu\\Sigma_f(x) = 0$, ensuring
    fission production is identically zero outside the core.

    Parameters
    ----------
    mesh : UniformMesh1D
        Spatial grid.
    material_model : PiecewiseMaterialModel
        Piecewise material domain model.
    """

    mesh: UniformMesh1D
    material_model: PiecewiseMaterialModel

    def apply(self, interior_flux: np.ndarray) -> np.ndarray:
        """Compute fission source distribution $\\nu\\Sigma_f(x_i) \\phi_i$."""
        _, _, nsf_nodal, _ = self.material_model.sample_nodal_properties(self.mesh)
        nsf_interior = nsf_nodal[1:-1]
        return nsf_interior * interior_flux

    def total_production(self, interior_flux: np.ndarray) -> float:
        """Compute total fission production $\\int_0^L \\nu\\Sigma_f(x) \\phi(x)\\,dx$."""
        source = self.apply(interior_flux)
        return self.mesh.integrate(source)
