"""One-dimensional steady-state neutron diffusion reactor simulator.

A research-oriented scientific package for solving the one-energy-group
neutron diffusion eigenvalue problem in homogeneous and heterogeneous slab
geometries using conservative finite differences and power iteration.
"""

from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.boundary_conditions import DirichletZeroFluxBC
from neutron_diffusion.operators import LossOperator
from neutron_diffusion.solver import (
    PowerIterationSolver,
    SolverConfig,
    SolverResult,
    ConvergenceHistory,
)
from neutron_diffusion.analytics import (
    analytical_buckling,
    analytical_keff,
    analytical_fundamental_flux,
)
from neutron_diffusion.config import SimulationConfig

# Version 2 Heterogeneous additions
from neutron_diffusion.regions import (
    MaterialRegion,
    PiecewiseMaterialModel,
    create_reflected_slab,
)
from neutron_diffusion.heterogeneous_operators import (
    HeterogeneousLossOperator,
    HeterogeneousFissionOperator,
    harmonic_mean,
)
from neutron_diffusion.heterogeneous_solver import (
    HeterogeneousPowerIterationSolver,
    HeterogeneousSolverResult,
)

__version__ = "0.2.0"

__all__ = [
    # Version 1 Baseline
    "SlabGeometry",
    "MaterialProperties",
    "UniformMesh1D",
    "DirichletZeroFluxBC",
    "LossOperator",
    "PowerIterationSolver",
    "SolverConfig",
    "SolverResult",
    "ConvergenceHistory",
    "analytical_buckling",
    "analytical_keff",
    "analytical_fundamental_flux",
    "SimulationConfig",
    # Version 2 Heterogeneous Extensions
    "MaterialRegion",
    "PiecewiseMaterialModel",
    "create_reflected_slab",
    "HeterogeneousLossOperator",
    "HeterogeneousFissionOperator",
    "HeterogeneousPowerIterationSolver",
    "HeterogeneousSolverResult",
    "harmonic_mean",
]
