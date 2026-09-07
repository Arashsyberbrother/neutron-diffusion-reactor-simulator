"""Power iteration eigenvalue solver for steady-state 1D neutron diffusion."""

import math
import time
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

from neutron_diffusion.materials import MaterialProperties
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.operators import FissionOperator, LossOperator
from neutron_diffusion.metrics import normalize_flux


class ConvergenceError(RuntimeError):
    """Raised when the power iteration fails to converge within the maximum iterations."""
    pass


@dataclass
class SolverConfig:
    """Configuration options for power iteration eigenvalue solver.

    Parameters
    ----------
    tolerance_k : float, optional
        Convergence tolerance on eigenvalue difference $|k^{(m+1)} - k^{(m)}|$.
        Default is 1.0e-7.
    tolerance_flux : float, optional
        Convergence tolerance on relative flux $L_2$ difference.
        Default is 1.0e-6.
    max_iterations : int, optional
        Maximum allowable outer power iterations. Default is 1000.
    initial_flux_guess : {"uniform", "sinusoidal"}, optional
        Shape of initial flux trial function. Default is "uniform".
    flux_normalization : {"l2", "max", "integral"}, optional
        Normalization convention enforced on the final converged flux.
        Default is "l2".
    """

    tolerance_k: float = 1.0e-7
    tolerance_flux: float = 1.0e-6
    max_iterations: int = 1000
    initial_flux_guess: Literal["uniform", "sinusoidal"] = "uniform"
    flux_normalization: Literal["l2", "max", "integral"] = "l2"

    def __post_init__(self) -> None:
        if self.tolerance_k <= 0.0:
            raise ValueError(f"tolerance_k must be strictly positive, got {self.tolerance_k}")
        if self.tolerance_flux <= 0.0:
            raise ValueError(f"tolerance_flux must be strictly positive, got {self.tolerance_flux}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")


@dataclass
class ConvergenceHistory:
    """Record of iteration metrics across the power iteration trajectory.

    Attributes
    ----------
    iterations : list[int]
        Iteration indices $0, 1, \\dots, M$.
    keff : list[float]
        Eigenvalue estimate at each iteration.
    delta_keff : list[float]
        Absolute change $|k^{(m+1)} - k^{(m)}|$ at each iteration.
    flux_residual : list[float]
        Relative $L_2$ change in interior flux vector at each iteration.
    source_residual : list[float]
        Relative $L_2$ norm of the eigenvalue residual
        $\\|\\mathbf{A}\\boldsymbol{\\phi} - \\frac{1}{k}\\mathbf{F}\\boldsymbol{\\phi}\\|_2$.
    """

    iterations: list[int] = field(default_factory=list)
    keff: list[float] = field(default_factory=list)
    delta_keff: list[float] = field(default_factory=list)
    flux_residual: list[float] = field(default_factory=list)
    source_residual: list[float] = field(default_factory=list)

    @property
    def delta_k(self) -> list[float]:
        """Convenience alias for delta_keff."""
        return self.delta_keff

    def record(
        self,
        iteration: int,
        k: float,
        delta_k: float,
        flux_res: float,
        src_res: float,
    ) -> None:
        """Append an iteration step to history."""
        self.iterations.append(iteration)
        self.keff.append(float(k))
        self.delta_keff.append(float(delta_k))
        self.flux_residual.append(float(flux_res))
        self.source_residual.append(float(src_res))


@dataclass
class SolverResult:
    """Structured container for converged power iteration solution.

    Attributes
    ----------
    keff : float
        Converged effective multiplication factor $k_{\\text{eff}}$.
    flux : np.ndarray
        Converged scalar flux distribution across all mesh points $[0, L]$
        (length $N+1$), normalized according to configuration.
    interior_flux : np.ndarray
        Converged interior flux vector (length $N-1$).
    normalized_power : np.ndarray
        Normalized core power / fission source distribution across mesh nodes.
    iterations : int
        Total number of power iterations executed.
    converged : bool
        True if solver achieved requested tolerances.
    history : ConvergenceHistory
        Complete trajectory of eigenvalue and flux residuals.
    runtime_seconds : float
        Wall-clock time consumed by the solver in seconds.
    mesh : UniformMesh1D
        Underlying spatial grid.
    materials : MaterialProperties
        Material cross sections and diffusion properties.
    config : SolverConfig
        Configuration parameters used during solve.
    """

    keff: float
    flux: np.ndarray
    interior_flux: np.ndarray
    normalized_power: np.ndarray
    iterations: int
    converged: bool
    history: ConvergenceHistory
    runtime_seconds: float
    mesh: UniformMesh1D
    materials: MaterialProperties
    config: SolverConfig


class PowerIterationSolver:
    """Primary eigenvalue solver implementing the power iteration method for 1D diffusion.

    Mathematical formulation:
    At each outer iteration $m$:
    1. Update fission source:
       $$\\mathbf{s}^{(m)} = \\frac{1}{k^{(m)}} \\mathbf{F} \\boldsymbol{\\phi}^{(m)}$$
    2. Invert loss operator via tridiagonal solve:
       $$\\mathbf{A} \\tilde{\\boldsymbol{\\phi}}^{(m+1)} = \\mathbf{s}^{(m)}$$
    3. Compute updated fission production and eigenvalue:
       $$k^{(m+1)} = k^{(m)} \\frac{\\int_0^L \\nu\\Sigma_f \\tilde{\\phi}^{(m+1)}\\,dx}{\\int_0^L \\nu\\Sigma_f \\phi^{(m)}\\,dx}$$
    4. Normalize flux:
       $$\\boldsymbol{\\phi}^{(m+1)} = \\frac{\\tilde{\\boldsymbol{\\phi}}^{(m+1)}}{\\|\\tilde{\\boldsymbol{\\phi}}^{(m+1)}\\|}$$
    5. Evaluate convergence criteria and record history.
    """

    def __init__(
        self,
        mesh: UniformMesh1D,
        materials: MaterialProperties,
        config: SolverConfig | None = None,
    ) -> None:
        self.mesh = mesh
        self.materials = materials
        self.config = config or SolverConfig()
        self.loss_operator = LossOperator(mesh=mesh, materials=materials)
        self.fission_operator = FissionOperator(mesh=mesh, materials=materials)

    def _generate_initial_flux(self) -> np.ndarray:
        """Create initial trial flux vector for interior nodes."""
        n = self.mesh.num_interior
        if self.config.initial_flux_guess == "uniform":
            flux = np.ones(n, dtype=np.float64)
        elif self.config.initial_flux_guess == "sinusoidal":
            x_int = self.mesh.x_interior
            flux = np.sin(math.pi * x_int / self.mesh.geometry.length)
        else:
            raise ValueError(f"Unknown initial flux guess: {self.config.initial_flux_guess}")

        # Normalize initial guess using continuous L2 norm
        full = self.mesh.reconstruct_full_field(flux, 0.0, 0.0)
        norm = math.sqrt(self.mesh.integrate(full ** 2))
        return flux / norm

    def solve(self) -> SolverResult:
        """Execute power iteration until convergence or max iterations.

        Returns
        -------
        SolverResult
            Structured result containing converged $k_{\\text{eff}}$, flux, and metrics.

        Raises
        ------
        ConvergenceError
            If convergence tolerances are not met within `config.max_iterations`.
        """
        start_time = time.perf_counter()

        # Initial state
        phi_m = self._generate_initial_flux()
        k_m = 1.0
        history = ConvergenceHistory()

        converged = False
        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1

            # 1. Construct fission source vector
            fission_source = (1.0 / k_m) * self.fission_operator.apply(phi_m)

            # 2. Solve discrete diffusion equation A * phi_candidate = fission_source
            phi_candidate = self.loss_operator.solve(fission_source)

            # Ensure non-negativity (physical requirement)
            if np.any(phi_candidate < 0.0):
                # In fundamental mode with positive source and M-matrix A, all elements are strictly positive
                phi_candidate = np.maximum(phi_candidate, 0.0)

            # 3. Compute new total fission production and update k_eff
            production_new = self.fission_operator.total_production(phi_candidate)
            production_old = self.fission_operator.total_production(phi_m)

            if production_old == 0.0:
                raise RuntimeError("Previous iteration fission production is zero.")

            k_next = k_m * (production_new / production_old)

            # 4. Normalize candidate flux for next iteration
            full_cand = self.mesh.reconstruct_full_field(phi_candidate, 0.0, 0.0)
            cand_l2 = math.sqrt(self.mesh.integrate(full_cand ** 2))
            if cand_l2 == 0.0:
                raise RuntimeError("Candidate flux norm is zero.")
            phi_next = phi_candidate / cand_l2

            # 5. Convergence metrics
            delta_k = abs(k_next - k_m)
            flux_diff_norm = math.sqrt(np.sum((phi_next - phi_m) ** 2))
            phi_norm = math.sqrt(np.sum(phi_next ** 2))
            rel_flux_change = flux_diff_norm / phi_norm if phi_norm > 0.0 else float("inf")

            # Source residual: ||A*phi - (1/k)*F*phi||_2 / ||(1/k)*F*phi||_2
            a_phi = self.loss_operator.apply(phi_next)
            f_phi = (1.0 / k_next) * self.fission_operator.apply(phi_next)
            res_vector = a_phi - f_phi
            src_norm = math.sqrt(np.sum(f_phi ** 2))
            source_residual = math.sqrt(np.sum(res_vector ** 2)) / src_norm if src_norm > 0.0 else 0.0

            history.record(
                iteration=iteration,
                k=k_next,
                delta_k=delta_k,
                flux_res=rel_flux_change,
                src_res=source_residual,
            )

            # 6. Check convergence criteria
            if delta_k < self.config.tolerance_k and rel_flux_change < self.config.tolerance_flux:
                converged = True
                phi_m = phi_next
                k_m = k_next
                break

            # Advance to next iteration
            phi_m = phi_next
            k_m = k_next

        elapsed = time.perf_counter() - start_time

        if not converged:
            raise ConvergenceError(
                f"Power iteration failed to converge in {self.config.max_iterations} iterations. "
                f"Last delta_k = {delta_k:.2e} (tol={self.config.tolerance_k:.2e}), "
                f"relative flux change = {rel_flux_change:.2e} (tol={self.config.tolerance_flux:.2e})."
            )

        # Final reconstruction and chosen normalization
        full_flux = self.mesh.reconstruct_full_field(phi_m, 0.0, 0.0)
        final_flux = normalize_flux(full_flux, self.mesh, method=self.config.flux_normalization)
        final_interior = final_flux[1:-1]

        # Normalized power distribution:
        # In a homogeneous 1-group slab, fission rate Sigma_f * phi(x) is proportional to phi(x).
        # Power is normalized such that its spatial average is 1.0 (or peak 1.0, here integral / L = 1.0)
        integrated_power = self.mesh.integrate(final_flux)
        normalized_power = (
            (final_flux / (integrated_power / self.mesh.geometry.length))
            if integrated_power > 0.0
            else final_flux
        )

        return SolverResult(
            keff=k_m,
            flux=final_flux,
            interior_flux=final_interior,
            normalized_power=normalized_power,
            iterations=iteration,
            converged=converged,
            history=history,
            runtime_seconds=elapsed,
            mesh=self.mesh,
            materials=self.materials,
            config=self.config,
        )
