"""Spatial discretization and uniform 1D mesh representations."""

from dataclasses import dataclass
import numpy as np

from neutron_diffusion.geometry import SlabGeometry


@dataclass(frozen=True)
class UniformMesh1D:
    """Uniform one-dimensional spatial grid for finite-difference discretization.

    Parameters
    ----------
    geometry : SlabGeometry
        Slab domain specification containing length $L$.
    num_cells : int
        Number of spatial cells (intervals) $N$. Must be $\\ge 2$.

    Attributes
    ----------
    dx : float
        Uniform grid spacing $\\Delta x = L / N$ in cm.
    x : np.ndarray
        Array of all $N + 1$ nodal coordinates from $x_0 = 0$ to $x_N = L$.
    x_interior : np.ndarray
        Array of $N - 1$ interior nodal coordinates from $x_1$ to $x_{N-1}$.
    num_points : int
        Total number of nodes $N + 1$.
    num_interior : int
        Number of interior unknowns $N - 1$.
    """

    geometry: SlabGeometry
    num_cells: int

    def __post_init__(self) -> None:
        if self.num_cells < 2:
            raise ValueError(
                f"Number of mesh cells must be at least 2 for interior discretization, "
                f"got {self.num_cells}."
            )

    @property
    def dx(self) -> float:
        """Uniform mesh spacing $\\Delta x$ in cm."""
        return self.geometry.length / float(self.num_cells)

    @property
    def num_points(self) -> int:
        """Total number of grid nodes including boundaries."""
        return self.num_cells + 1

    @property
    def num_interior(self) -> int:
        """Number of interior grid nodes."""
        return self.num_cells - 1

    @property
    def x(self) -> np.ndarray:
        """Full array of nodal coordinates from $x=0$ to $x=L$."""
        return np.linspace(0.0, self.geometry.length, self.num_points)

    @property
    def x_interior(self) -> np.ndarray:
        """Array of interior nodal coordinates."""
        return self.x[1:-1]

    def integrate(self, field: np.ndarray) -> float:
        """Compute the spatial integral $\\int_0^L f(x)\\,dx$ using the composite trapezoidal rule.

        Parameters
        ----------
        field : np.ndarray
            Field values. Can be either full nodal array (length $N+1$)
            or interior array (length $N-1$, assuming zero at boundaries).

        Returns
        -------
        float
            Integrated value.
        """
        if len(field) == self.num_points:
            return float(np.trapezoid(field, self.x) if hasattr(np, "trapezoid") else np.trapz(field, self.x))
        elif len(field) == self.num_interior:
            full = self.reconstruct_full_field(field, 0.0, 0.0)
            return float(np.trapezoid(full, self.x) if hasattr(np, "trapezoid") else np.trapz(full, self.x))
        else:
            raise ValueError(
                f"Field length {len(field)} does not match full ({self.num_points}) "
                f"or interior ({self.num_interior}) mesh sizes."
            )

    def reconstruct_full_field(
        self,
        interior_field: np.ndarray,
        left_val: float = 0.0,
        right_val: float = 0.0,
    ) -> np.ndarray:
        """Reconstruct full $(N+1)$-point field from $(N-1)$ interior unknowns and boundary values.

        Parameters
        ----------
        interior_field : np.ndarray
            Values at interior nodes $x_1, \\dots, x_{N-1}$.
        left_val : float, optional
            Boundary value at $x = 0$. Default is 0.0.
        right_val : float, optional
            Boundary value at $x = L$. Default is 0.0.

        Returns
        -------
        np.ndarray
            Full nodal array of shape $(N+1,)$.
        """
        if len(interior_field) != self.num_interior:
            raise ValueError(
                f"Expected interior field of length {self.num_interior}, got {len(interior_field)}."
            )
        full = np.empty(self.num_points, dtype=np.float64)
        full[0] = left_val
        full[1:-1] = interior_field
        full[-1] = right_val
        return full
