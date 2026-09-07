"""Boundary condition representations for 1D neutron diffusion problems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class BoundaryCondition(ABC):
    """Abstract base class for reactor boundary conditions."""

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the boundary condition."""
        pass


@dataclass(frozen=True)
class DirichletZeroFluxBC(BoundaryCondition):
    """Homogeneous Dirichlet zero-flux boundary condition: $\\phi(0) = 0$ and $\\phi(L) = 0$.

    In reactor physics, this represents the standard simplified vacuum boundary
    approximation where the neutron flux vanishes at the bare core outer surfaces
    (or at extrapolated boundaries $L + 2d$ where $d \\approx 0.71 \\lambda_{\\text{tr}}$).
    For the baseline benchmark, the physical boundary is taken directly as $x=0$ and $x=L$.
    """

    left_value: float = 0.0
    right_value: float = 0.0

    def description(self) -> str:
        return f"Zero-flux Dirichlet BC: phi(0) = {self.left_value}, phi(L) = {self.right_value}"
