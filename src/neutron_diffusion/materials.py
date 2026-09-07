"""Macroscopic cross sections and diffusion material property representations."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialProperties:
    """One-energy-group macroscopic cross sections and diffusion properties.

    Parameters
    ----------
    D : float
        Neutron diffusion coefficient $D$ in centimeters (cm).
        Must be strictly positive ($D > 0$).
    sigma_a : float
        Macroscopic absorption cross section $\\Sigma_a$ in $\\text{cm}^{-1}$.
        Must be non-negative ($\\Sigma_a \\ge 0$).
    nu_sigma_f : float
        Macroscopic neutron production cross section $\\nu\\Sigma_f$ in $\\text{cm}^{-1}$.
        Must be non-negative ($\\nu\\Sigma_f \\ge 0$).
    name : str, optional
        Human-readable descriptor for the material medium. Default is
        "Homogeneous Core Medium".

    Raises
    ------
    ValueError
        If physical constraints on cross sections or diffusion parameters are violated.
    """

    D: float
    sigma_a: float
    nu_sigma_f: float
    name: str = "Homogeneous Core Medium"

    def __post_init__(self) -> None:
        if self.D <= 0.0:
            raise ValueError(f"Diffusion coefficient D must be strictly positive, got {self.D} cm.")
        if self.sigma_a < 0.0:
            raise ValueError(f"Absorption cross section Sigma_a must be non-negative, got {self.sigma_a} cm^-1.")
        if self.nu_sigma_f < 0.0:
            raise ValueError(f"Fission production cross section nuSigma_f must be non-negative, got {self.nu_sigma_f} cm^-1.")

    @property
    def diffusion_length(self) -> float:
        """Thermal diffusion length $L_d = \\sqrt{D / \\Sigma_a}$ in centimeters.

        Returns
        -------
        float
            Diffusion length $L_d$. Returns infinity if $\\Sigma_a = 0$.
        """
        if self.sigma_a == 0.0:
            return float("inf")
        return math.sqrt(self.D / self.sigma_a)

    @property
    def k_inf(self) -> float:
        """Infinite-medium multiplication factor $k_\\infty = \\nu\\Sigma_f / \\Sigma_a$.

        Returns
        -------
        float
            Multiplication factor for an infinite medium without leakage.
            Returns infinity if $\\Sigma_a = 0$ and $\\nu\\Sigma_f > 0$.
        """
        if self.sigma_a == 0.0:
            return float("inf") if self.nu_sigma_f > 0.0 else 0.0
        return self.nu_sigma_f / self.sigma_a

    @property
    def material_buckling(self) -> float:
        """Material buckling $B_m^2 = (\\nu\\Sigma_f - \\Sigma_a) / D$ in $\\text{cm}^{-2}$.

        Returns
        -------
        float
            Material buckling characterising medium multiplying capacity.
        """
        return (self.nu_sigma_f - self.sigma_a) / self.D
