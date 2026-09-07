"""Configuration structures and standardized benchmark parameter presets."""

from dataclasses import dataclass
from typing import Literal

from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.solver import SolverConfig


@dataclass(frozen=True)
class SimulationConfig:
    """Master simulation configuration encapsulating physics, mesh, and solver settings.

    Parameters
    ----------
    length : float, optional
        Slab width $L$ in cm. Default is 100.0 cm.
    D : float, optional
        Diffusion coefficient $D$ in cm. Default is 1.0 cm.
    sigma_a : float, optional
        Absorption cross section $\\Sigma_a$ in $\\text{cm}^{-1}$. Default is 0.02 $\\text{cm}^{-1}$.
    nu_sigma_f : float, optional
        Neutron production cross section $\\nu\\Sigma_f$ in $\\text{cm}^{-1}$. Default is 0.025 $\\text{cm}^{-1}$.
    num_cells : int, optional
        Number of spatial grid intervals $N$. Default is 100.
    tolerance_k : float, optional
        Eigenvalue convergence tolerance. Default is 1.0e-7.
    tolerance_flux : float, optional
        Flux $L_2$ relative convergence tolerance. Default is 1.0e-6.
    max_iterations : int, optional
        Maximum power iterations. Default is 1000.
    initial_flux_guess : {"uniform", "sinusoidal"}, optional
        Initial flux guess shape. Default is "uniform".
    flux_normalization : {"l2", "max", "integral"}, optional
        Flux normalization convention. Default is "l2".
    """

    length: float = 100.0
    D: float = 1.0
    sigma_a: float = 0.02
    nu_sigma_f: float = 0.025
    num_cells: int = 100
    tolerance_k: float = 1.0e-7
    tolerance_flux: float = 1.0e-6
    max_iterations: int = 1000
    initial_flux_guess: Literal["uniform", "sinusoidal"] = "uniform"
    flux_normalization: Literal["l2", "max", "integral"] = "l2"

    def to_geometry(self) -> SlabGeometry:
        """Create configured SlabGeometry instance."""
        return SlabGeometry(length=self.length)

    def to_materials(self, name: str = "Homogeneous Core Medium") -> MaterialProperties:
        """Create configured MaterialProperties instance."""
        return MaterialProperties(
            D=self.D,
            sigma_a=self.sigma_a,
            nu_sigma_f=self.nu_sigma_f,
            name=name,
        )

    def to_mesh(self) -> UniformMesh1D:
        """Create configured UniformMesh1D instance."""
        return UniformMesh1D(geometry=self.to_geometry(), num_cells=self.num_cells)

    def to_solver_config(self) -> SolverConfig:
        """Create configured SolverConfig instance."""
        return SolverConfig(
            tolerance_k=self.tolerance_k,
            tolerance_flux=self.tolerance_flux,
            max_iterations=self.max_iterations,
            initial_flux_guess=self.initial_flux_guess,
            flux_normalization=self.flux_normalization,
        )


def get_baseline_config() -> SimulationConfig:
    """Return standard baseline simulation configuration (supercritical benchmark, k_eff ~ 1.1912)."""
    return SimulationConfig()


def get_critical_config() -> SimulationConfig:
    """Return critically tuned configuration where nuSigma_f = Sigma_a + D*(pi/L)^2 (k_eff = 1.0000)."""
    # For L=100 cm, D=1.0 cm, Sigma_a=0.02 cm^-1:
    # Bg^2 = (pi/100)^2 = 9.8696044e-4 cm^-2
    # Critical nuSigma_f = 0.02 + 1.0 * 9.8696044e-4 = 0.02098696044 cm^-1
    crit_nu_sigma_f = 0.02 + 1.0 * (3.141592653589793 / 100.0) ** 2
    return SimulationConfig(nu_sigma_f=crit_nu_sigma_f)


def get_subcritical_config() -> SimulationConfig:
    """Return subcritical benchmark configuration (nuSigma_f = 0.015 cm^-1, k_eff ~ 0.7147)."""
    return SimulationConfig(nu_sigma_f=0.015)
