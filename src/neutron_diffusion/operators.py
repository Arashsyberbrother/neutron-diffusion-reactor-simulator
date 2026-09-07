"""Discrete differential operators and linear solvers for 1D neutron diffusion."""

from dataclasses import dataclass
import numpy as np

from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D


def solve_tridiagonal_thomas(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve tridiagonal linear system $A x = d$ using the Thomas algorithm (TDMA).

    The system is:
    $a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i$

    Parameters
    ----------
    lower : np.ndarray
        Sub-diagonal elements $a_1, \\dots, a_{n-1}$ (length $n-1$).
    diag : np.ndarray
        Main diagonal elements $b_0, \\dots, b_{n-1}$ (length $n$).
    upper : np.ndarray
        Super-diagonal elements $c_0, \\dots, c_{n-2}$ (length $n-1$).
    rhs : np.ndarray
        Right-hand side vector $d_0, \\dots, d_{n-1}$ (length $n$).

    Returns
    -------
    np.ndarray
        Solution vector $x$ of length $n$.

    Raises
    ------
    ZeroDivisionError
        If a zero pivot is encountered during elimination.
    """
    n = len(diag)
    if len(rhs) != n:
        raise ValueError(f"RHS length ({len(rhs)}) does not match matrix size ({n}).")
    if len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError(f"Lower ({len(lower)}) and upper ({len(upper)}) bands must be length {n-1}.")

    # Work on copies to prevent side effects
    c_prime = np.empty(n - 1, dtype=np.float64)
    d_prime = np.empty(n, dtype=np.float64)

    # Forward sweep
    b0 = diag[0]
    if b0 == 0.0:
        raise ZeroDivisionError("Zero pivot encountered at index 0 in Thomas algorithm.")
    c_prime[0] = upper[0] / b0
    d_prime[0] = rhs[0] / b0

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if denom == 0.0:
            raise ZeroDivisionError(f"Zero pivot encountered at index {i} in Thomas algorithm.")
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    denom_last = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
    if denom_last == 0.0:
        raise ZeroDivisionError(f"Zero pivot encountered at index {n-1} in Thomas algorithm.")
    d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom_last

    # Back substitution
    x = np.empty(n, dtype=np.float64)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


@dataclass(frozen=True)
class LossOperator:
    """Second-order finite difference discretization of the neutron loss operator:
    $\\mathcal{L}\\phi = -D \\frac{d^2\\phi}{dx^2} + \\Sigma_a \\phi$.

    On a uniform mesh with zero-flux Dirichlet boundaries $\\phi_0 = \\phi_N = 0$,
    the interior matrix $\\mathbf{A} \\in \\mathbb{R}^{(N-1) \\times (N-1)}$ is tridiagonal,
    symmetric positive definite (SPD), and strictly diagonally dominant.

    Parameters
    ----------
    mesh : UniformMesh1D
        Spatial grid discretization.
    materials : MaterialProperties
        Macroscopic cross sections and diffusion coefficient.
    """

    mesh: UniformMesh1D
    materials: MaterialProperties

    @property
    def num_interior(self) -> int:
        """Number of interior unknowns $n = N - 1$."""
        return self.mesh.num_interior

    @property
    def diagonal_value(self) -> float:
        """Main diagonal coefficient $b = \\frac{2D}{\\Delta x^2} + \\Sigma_a$."""
        dx = self.mesh.dx
        return (2.0 * self.materials.D) / (dx * dx) + self.materials.sigma_a

    @property
    def off_diagonal_value(self) -> float:
        """Off-diagonal coefficient $a = c = -\\frac{D}{\\Delta x^2}$."""
        dx = self.mesh.dx
        return -self.materials.D / (dx * dx)

    @property
    def bands(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tridiagonal bands (lower, main_diag, upper)."""
        n = self.num_interior
        lower = np.full(n - 1, self.off_diagonal_value, dtype=np.float64)
        diag = np.full(n, self.diagonal_value, dtype=np.float64)
        upper = np.full(n - 1, self.off_diagonal_value, dtype=np.float64)
        return lower, diag, upper

    def apply(self, interior_flux: np.ndarray) -> np.ndarray:
        """Matrix-vector multiplication $\\mathbf{A} \\boldsymbol{\\phi}$.

        Parameters
        ----------
        interior_flux : np.ndarray
            Interior flux vector of length $N-1$.

        Returns
        -------
        np.ndarray
            Result of $\\mathbf{A} \\boldsymbol{\\phi}$.
        """
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
        """Solve the fixed-source diffusion equation $\\mathbf{A} \\boldsymbol{\\phi} = \\mathbf{s}$.

        Uses the $O(N)$ tridiagonal Thomas algorithm.

        Parameters
        ----------
        source : np.ndarray
            Right-hand side source vector of length $N-1$.

        Returns
        -------
        np.ndarray
            Interior flux vector of length $N-1$.
        """
        lower, diag, upper = self.bands
        return solve_tridiagonal_thomas(lower, diag, upper, source)

    def to_dense(self) -> np.ndarray:
        """Construct full dense $(N-1) \\times (N-1)$ matrix for inspection and testing."""
        n = self.num_interior
        matrix = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(matrix, self.diagonal_value)
        np.fill_diagonal(matrix[:-1, 1:], self.off_diagonal_value)
        np.fill_diagonal(matrix[1:, :-1], self.off_diagonal_value)
        return matrix


@dataclass(frozen=True)
class FissionOperator:
    """Neutron fission production operator $\\mathcal{F}\\phi = \\nu\\Sigma_f \\phi$.

    Parameters
    ----------
    mesh : UniformMesh1D
        Spatial grid.
    materials : MaterialProperties
        Macroscopic cross sections.
    """

    mesh: UniformMesh1D
    materials: MaterialProperties

    def apply(self, interior_flux: np.ndarray) -> np.ndarray:
        """Compute fission source distribution $\\nu\\Sigma_f \\boldsymbol{\\phi}$."""
        return self.materials.nu_sigma_f * interior_flux

    def total_production(self, interior_flux: np.ndarray) -> float:
        """Compute total fission neutron production $\\int_0^L \\nu\\Sigma_f \\phi(x)\\,dx$."""
        source = self.apply(interior_flux)
        return self.mesh.integrate(source)
