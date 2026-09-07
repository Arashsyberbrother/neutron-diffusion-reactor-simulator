"""Error metrics, flux normalization, and numerical convergence rate estimation."""

import math
from typing import Literal
import numpy as np

from neutron_diffusion.mesh import UniformMesh1D


def normalize_flux(
    flux: np.ndarray,
    mesh: UniformMesh1D,
    method: Literal["l2", "max", "integral"] = "l2",
) -> np.ndarray:
    """Normalize a scalar neutron flux vector according to a chosen metric.

    Parameters
    ----------
    flux : np.ndarray
        Nodal flux array of length $N+1$ or interior flux of length $N-1$.
    mesh : UniformMesh1D
        Spatial grid.
    method : {"l2", "max", "integral"}, optional
        Normalization convention:
        - "l2": Continuous $L_2$ norm $\\sqrt{\\int_0^L \\phi(x)^2\\,dx} = 1.0$.
        - "max": Peak flux $\\max_x |\\phi(x)| = 1.0$.
        - "integral": Total integral $\\int_0^L \\phi(x)\\,dx = 1.0$.
        Default is "l2".

    Returns
    -------
    np.ndarray
        Normalized flux vector of identical shape and orientation.

    Raises
    ------
    ValueError
        If the flux norm is zero or method is unrecognised.
    """
    if method == "max":
        peak = float(np.max(np.abs(flux)))
        if peak == 0.0:
            raise ValueError("Cannot normalize zero flux vector.")
        return flux / peak

    elif method == "integral":
        integral = mesh.integrate(flux)
        if integral == 0.0:
            raise ValueError("Cannot normalize flux with zero spatial integral.")
        return flux / integral

    elif method == "l2":
        # Continuous L2 integral norm via composite trapezoidal integration
        if len(flux) == mesh.num_interior:
            full = mesh.reconstruct_full_field(flux, 0.0, 0.0)
        else:
            full = flux
        l2_norm = math.sqrt(mesh.integrate(full ** 2))
        if l2_norm == 0.0:
            raise ValueError("Cannot normalize flux with zero L2 norm.")
        return flux / l2_norm

    else:
        raise ValueError(f"Unknown normalization method '{method}'.")


def compute_keff_errors(
    numerical_keff: float,
    analytical_keff: float,
) -> dict[str, float]:
    """Calculate absolute and relative errors between numerical and analytical eigenvalues.

    Parameters
    ----------
    numerical_keff : float
        Eigenvalue obtained from numerical simulation.
    analytical_keff : float
        Exact reference eigenvalue.

    Returns
    -------
    dict[str, float]
        Dictionary with 'absolute_error', 'relative_error', and 'pcm_error' ($10^{-5} \\Delta k$).
    """
    abs_err = abs(numerical_keff - analytical_keff)
    rel_err = abs_err / analytical_keff
    pcm_err = abs_err * 1.0e5  # percent mille (pcm) standard in nuclear engineering
    return {
        "absolute_error": float(abs_err),
        "relative_error": float(rel_err),
        "pcm_error": float(pcm_err),
    }


def compute_flux_errors(
    numerical_flux: np.ndarray,
    analytical_flux: np.ndarray,
    mesh: UniformMesh1D,
) -> dict[str, float]:
    """Calculate pointwise and functional $L_2$ errors between normalized fluxes.

    Parameters
    ----------
    numerical_flux : np.ndarray
        Numerical flux array on full mesh (length $N+1$).
    analytical_flux : np.ndarray
        Analytical reference flux on full mesh (length $N+1$).
    mesh : UniformMesh1D
        Spatial grid.

    Returns
    -------
    dict[str, float]
        Dictionary with 'l2_absolute_error', 'l2_relative_error', and 'linf_error'.
    """
    if len(numerical_flux) != mesh.num_points or len(analytical_flux) != mesh.num_points:
        raise ValueError(
            f"Flux lengths ({len(numerical_flux)}, {len(analytical_flux)}) "
            f"must match mesh nodal size ({mesh.num_points})."
        )

    diff = numerical_flux - analytical_flux
    l2_abs = math.sqrt(mesh.integrate(diff ** 2))
    l2_ref = math.sqrt(mesh.integrate(analytical_flux ** 2))
    l2_rel = l2_abs / l2_ref if l2_ref > 0.0 else float("inf")
    linf_err = float(np.max(np.abs(diff)))

    return {
        "l2_absolute_error": float(l2_abs),
        "l2_relative_error": float(l2_rel),
        "linf_error": float(linf_err),
    }


def estimate_convergence_order(
    dx_values: list[float] | np.ndarray,
    error_values: list[float] | np.ndarray,
) -> tuple[float, float]:
    """Estimate the observed asymptotic order of convergence $p$ via linear regression.

    Fits the power law $\\text{Error} = C (\\Delta x)^p$, linearized as:
    $$\\ln(\\text{Error}) = p \\ln(\\Delta x) + \\ln(C)$$

    Parameters
    ----------
    dx_values : list[float] or np.ndarray
        Sequence of mesh spacing sizes $\\Delta x$.
    error_values : list[float] or np.ndarray
        Corresponding discretization errors.

    Returns
    -------
    tuple[float, float]
        Estimated convergence rate $p$ and regression coefficient $R^2$.
    """
    dx = np.asarray(dx_values, dtype=np.float64)
    err = np.asarray(error_values, dtype=np.float64)

    if len(dx) < 2:
        raise ValueError("At least 2 data points required to estimate convergence order.")

    log_dx = np.log(dx)
    log_err = np.log(err)

    # Linear least squares fit
    slope, intercept = np.polyfit(log_dx, log_err, 1)

    # Compute coefficient of determination R^2
    predicted = slope * log_dx + intercept
    ss_tot = float(np.sum((log_err - np.mean(log_err)) ** 2))
    ss_res = float(np.sum((log_err - predicted) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0

    return float(slope), float(r_squared)
