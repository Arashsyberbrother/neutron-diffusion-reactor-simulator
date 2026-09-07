"""Analytical benchmark solutions and exact eigenvalue derivations for 1D homogeneous slab."""

import math
import numpy as np

from neutron_diffusion.geometry import SlabGeometry
from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D


def analytical_geometric_buckling(geometry: SlabGeometry, mode: int = 1) -> float:
    """Compute the analytical geometric buckling $B_g^2$ for a 1D slab with zero-flux boundaries.

    For mode $n$, the geometric buckling is:
    $$B_{g,n}^2 = \\left(\\frac{n \\pi}{L}\\right)^2$$

    Parameters
    ----------
    geometry : SlabGeometry
        Slab geometry containing core thickness $L$.
    mode : int, optional
        Harmonic mode number $n \\ge 1$. Default is 1 (fundamental mode).

    Returns
    -------
    float
        Geometric buckling $B_g^2$ in $\\text{cm}^{-2}$.

    Raises
    ------
    ValueError
        If harmonic mode $n < 1$.
    """
    if mode < 1:
        raise ValueError(f"Harmonic mode must be >= 1, got {mode}.")
    return (float(mode) * math.pi / geometry.length) ** 2


# Alias for concise reference
analytical_buckling = analytical_geometric_buckling


def analytical_keff(
    materials: MaterialProperties,
    geometry: SlabGeometry,
    mode: int = 1,
) -> float:
    """Compute the exact analytical effective multiplication factor $k_{\\text{eff}}$.

    For a homogeneous slab reactor with zero-flux boundary conditions, the fundamental
    eigenmode is $\\phi(x) = \\sin(\\pi x / L)$. Substituting into the 1D neutron diffusion
    eigenvalue equation:
    $$-D \\frac{d^2\\phi}{dx^2} + \\Sigma_a \\phi = \\frac{1}{k_{\\text{eff}}} \\nu\\Sigma_f \\phi$$
    yields the fundamental-mode analytical eigenvalue:
    $$k_{\\text{eff,analytical}} = \\frac{\\nu\\Sigma_f}{\\Sigma_a + D B_g^2}$$
    where
    $$B_g^2 = \\left(\\frac{\\pi}{L}\\right)^2$$

    Parameters
    ----------
    materials : MaterialProperties
        Macroscopic material properties ($D$, $\\Sigma_a$, $\\nu\\Sigma_f$).
    geometry : SlabGeometry
        Slab geometry containing thickness $L$.
    mode : int, optional
        Spatial harmonic mode $n \\ge 1$. Default is 1 (fundamental mode).

    Returns
    -------
    float
        Analytical eigenvalue $k_{\\text{eff}}$.

    Raises
    ------
    ZeroDivisionError
        If total removal plus leakage $\\Sigma_a + D B_g^2 = 0$.
    """
    bg_sq = analytical_geometric_buckling(geometry, mode=mode)
    denom = materials.sigma_a + materials.D * bg_sq
    if denom == 0.0:
        raise ZeroDivisionError("Total absorption plus leakage denominator is zero.")
    return materials.nu_sigma_f / denom


def analytical_critical_production(
    materials: MaterialProperties,
    geometry: SlabGeometry,
) -> float:
    """Compute the critical neutron production cross section $\\nu\\Sigma_f$ for $k_{\\text{eff}} = 1$.

    From the analytical eigenvalue relation:
    $$k_{\\text{eff}} = \\frac{\\nu\\Sigma_f}{\\Sigma_a + D B_g^2} = 1.0$$
    it follows that criticality requires:
    $$\\nu\\Sigma_f = \\Sigma_a + D B_g^2$$

    Parameters
    ----------
    materials : MaterialProperties
        Material properties defining $D$ and $\\Sigma_a$.
    geometry : SlabGeometry
        Slab geometry defining $L$.

    Returns
    -------
    float
        Critical production cross section $\\nu\\Sigma_{f,\\text{crit}}$ in $\\text{cm}^{-1}$.
    """
    bg_sq = analytical_geometric_buckling(geometry, mode=1)
    return materials.sigma_a + materials.D * bg_sq


def analytical_fundamental_flux(
    mesh: UniformMesh1D,
    normalization: str = "l2",
) -> np.ndarray:
    """Compute the analytical fundamental-mode eigenfunction $\\phi_1(x) = \\sin(\\pi x / L)$.

    Parameters
    ----------
    mesh : UniformMesh1D
        Spatial grid on which the eigenfunction is evaluated.
    normalization : str, optional
        Flux normalization convention:
        - "l2": Continuous $L_2$ integral normalization, $\\sqrt{\\int_0^L \\phi(x)^2\\,dx} = 1$.
        - "max": Peak flux normalization, $\\max_x \\phi(x) = 1$.
        - "integral": Total integral normalization, $\\int_0^L \\phi(x)\\,dx = 1$.
        Default is "l2".

    Returns
    -------
    np.ndarray
        Analytical flux evaluated at all grid nodes $x \\in [0, L]$, shape $(N+1,)$.
    """
    x = mesh.x
    length = mesh.geometry.length
    unnorm = np.sin(math.pi * x / length)

    if normalization == "max":
        return unnorm / np.max(unnorm)
    elif normalization == "integral":
        total = mesh.integrate(unnorm)
        return unnorm / total
    elif normalization == "l2":
        # Analytical integral of sin^2(pi*x/L) over [0, L] is L/2
        norm_factor = math.sqrt(2.0 / length)
        return unnorm * norm_factor
    else:
        raise ValueError(
            f"Unknown normalization '{normalization}'. Supported: 'l2', 'max', 'integral'."
        )
