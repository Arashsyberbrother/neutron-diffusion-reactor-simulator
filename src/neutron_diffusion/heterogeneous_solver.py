"""Power iteration eigenvalue solver for multi-region heterogeneous 1D slabs."""

import math
import time
from dataclasses import dataclass
import numpy as np

from neutron_diffusion.heterogeneous_operators import (
    HeterogeneousFissionOperator,
    HeterogeneousLossOperator,
)
from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.metrics import normalize_flux
from neutron_diffusion.regions import PiecewiseMaterialModel
from neutron_diffusion.solver import ConvergenceError, ConvergenceHistory, SolverConfig


@dataclass
class HeterogeneousSolverResult:
    """Container for converged multi-region heterogeneous eigenvalue solution.

    Attributes
    ----------
    keff : float
        Converged effective multiplication factor $k_{\\text{eff}}$.
    flux : np.ndarray
        Converged scalar neutron flux across entire domain (length $N+1$).
    interior_flux : np.ndarray
        Converged interior nodal flux vector (length $N-1$).
    normalized_power : np.ndarray
        Normalized fission power distribution (strictly zero in reflector, non-zero in core).
    face_currents : np.ndarray
        Net neutron currents $J = -D \\frac{d\\phi}{dx}$ at cell faces (length $N$).
    iterations : int
        Total power iterations executed.
    converged : bool
        True if convergence criteria were satisfied.
    history : ConvergenceHistory
        Trajectory of eigenvalue and residuals.
    runtime_seconds : float
        Wall-clock elapsed solve time in seconds.
    mesh : UniformMesh1D
        Spatial grid.
    material_model : PiecewiseMaterialModel
        Piecewise material domain model.
    config : SolverConfig
        Solver configuration parameters.
    """

    keff: float
    flux: np.ndarray
    interior_flux: np.ndarray
    normalized_power: np.ndarray
    face_currents: np.ndarray
    iterations: int
    converged: bool
    history: ConvergenceHistory
    runtime_seconds: float
    mesh: UniformMesh1D
    material_model: PiecewiseMaterialModel
    config: SolverConfig

    def evaluate_interface_balance(self, interface_x: float) -> dict[str, float]:
        """Evaluate scalar flux and neutron current continuity at a material interface.

        Both derivatives and currents are defined with the standard positive $x$-direction convention:
        $$J(x) = -D(x) \\frac{d\\phi}{dx}$$
        Physical interface continuity requires:
        $$\\phi(x_{\\text{int}}^-) = \\phi(x_{\\text{int}}^+)$$
        $$J(x_{\\text{int}}^-) = J(x_{\\text{int}}^+)$$

        Parameters
        ----------
        interface_x : float
            Coordinate of material interface in cm.

        Returns
        -------
        dict[str, float]
            Dictionary containing left and right fluxes, left and right currents,
            and absolute differences across the interface.
        """
        mesh = self.mesh
        dx = mesh.dx
        # Locate nearest node or cell face
        node_idx = int(round(interface_x / dx))
        node_idx = max(1, min(node_idx, mesh.num_points - 2))

        phi_left = float(self.flux[node_idx - 1])
        phi_center = float(self.flux[node_idx])
        phi_right = float(self.flux[node_idx + 1])

        # Currents at adjacent faces around interface node
        j_left_face = float(self.face_currents[node_idx - 1])
        j_right_face = float(self.face_currents[node_idx])

        return {
            "interface_x": interface_x,
            "phi_node": phi_center,
            "phi_left_neighbor": phi_left,
            "phi_right_neighbor": phi_right,
            "flux_jump": abs(phi_center - 0.5 * (phi_left + phi_right)),
            "current_left_face": j_left_face,
            "current_right_face": j_right_face,
            "current_difference": abs(j_right_face - j_left_face),
        }


class HeterogeneousPowerIterationSolver:
    """Eigenvalue solver for multi-region heterogeneous 1D diffusion problems."""

    def __init__(
        self,
        mesh: UniformMesh1D,
        material_model: PiecewiseMaterialModel,
        config: SolverConfig | None = None,
    ) -> None:
        self.mesh = mesh
        self.material_model = material_model
        self.config = config or SolverConfig()
        self.loss_operator = HeterogeneousLossOperator(mesh=mesh, material_model=material_model)
        self.fission_operator = HeterogeneousFissionOperator(mesh=mesh, material_model=material_model)

    def _generate_initial_flux(self) -> np.ndarray:
        """Create initial positive trial flux vector."""
        n = self.mesh.num_interior
        if self.config.initial_flux_guess == "uniform":
            flux = np.ones(n, dtype=np.float64)
        elif self.config.initial_flux_guess == "sinusoidal":
            x_int = self.mesh.x_interior
            flux = np.sin(math.pi * x_int / self.mesh.geometry.length)
        else:
            raise ValueError(f"Unknown initial flux guess: {self.config.initial_flux_guess}")

        full = self.mesh.reconstruct_full_field(flux, 0.0, 0.0)
        norm = math.sqrt(self.mesh.integrate(full ** 2))
        return flux / norm

    def solve(self) -> HeterogeneousSolverResult:
        """Execute power iteration until convergence or max iterations reached.

        Returns
        -------
        HeterogeneousSolverResult
            Structured result containing converged $k_{\\text{eff}}$, flux, and metrics.
        """
        start_time = time.perf_counter()

        phi_m = self._generate_initial_flux()
        k_m = 1.0
        history = ConvergenceHistory()

        converged = False
        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1

            # 1. Update fission source vector (zero outside fuel)
            fission_source = (1.0 / k_m) * self.fission_operator.apply(phi_m)

            # 2. Solve discrete diffusion system A * phi_candidate = fission_source
            phi_candidate = self.loss_operator.solve(fission_source)

            # Enforce non-negativity
            if np.any(phi_candidate < 0.0):
                phi_candidate = np.maximum(phi_candidate, 0.0)

            # 3. Compute new total fission production and eigenvalue update
            prod_new = self.fission_operator.total_production(phi_candidate)
            prod_old = self.fission_operator.total_production(phi_m)

            if prod_old == 0.0:
                raise RuntimeError("Fission production in active core is zero.")

            k_next = k_m * (prod_new / prod_old)

            # 4. Normalize candidate flux for next iteration
            full_cand = self.mesh.reconstruct_full_field(phi_candidate, 0.0, 0.0)
            cand_l2 = math.sqrt(self.mesh.integrate(full_cand ** 2))
            if cand_l2 == 0.0:
                raise RuntimeError("Candidate flux norm is zero.")
            phi_next = phi_candidate / cand_l2

            # 5. Track convergence metrics
            delta_k = abs(k_next - k_m)
            diff_norm = math.sqrt(np.sum((phi_next - phi_m) ** 2))
            curr_norm = math.sqrt(np.sum(phi_next ** 2))
            rel_flux_change = diff_norm / curr_norm if curr_norm > 0.0 else float("inf")

            # Source residual
            a_phi = self.loss_operator.apply(phi_next)
            f_phi = (1.0 / k_next) * self.fission_operator.apply(phi_next)
            res_vec = a_phi - f_phi
            f_norm = math.sqrt(np.sum(f_phi ** 2))
            src_res = math.sqrt(np.sum(res_vec ** 2)) / f_norm if f_norm > 0.0 else 0.0

            history.record(
                iteration=iteration,
                k=k_next,
                delta_k=delta_k,
                flux_res=rel_flux_change,
                src_res=src_res,
            )

            # 6. Check tolerance
            if delta_k < self.config.tolerance_k and rel_flux_change < self.config.tolerance_flux:
                converged = True
                phi_m = phi_next
                k_m = k_next
                break

            phi_m = phi_next
            k_m = k_next

        elapsed = time.perf_counter() - start_time

        if not converged:
            raise ConvergenceError(
                f"Heterogeneous power iteration failed to converge in {self.config.max_iterations} iterations. "
                f"Last delta_k = {delta_k:.2e}, rel_flux_change = {rel_flux_change:.2e}."
            )

        # Final reconstruction and normalization
        full_flux = self.mesh.reconstruct_full_field(phi_m, 0.0, 0.0)
        final_flux = normalize_flux(full_flux, self.mesh, method=self.config.flux_normalization)
        final_interior = final_flux[1:-1]

        # Compute face currents
        face_currents = self.loss_operator.compute_face_currents(final_interior)

        # Fission power density: strictly non-zero only in fissioning regions
        _, _, nsf_nodal, fiss_mask = self.material_model.sample_nodal_properties(self.mesh)
        raw_power = np.zeros_like(final_flux)
        raw_power[fiss_mask] = final_flux[fiss_mask] * (nsf_nodal[fiss_mask] / np.max(nsf_nodal[fiss_mask]))

        # Normalize relative power such that average power density in the active fuel region is 1.0
        fuel_integral = self.mesh.integrate(raw_power)
        # Compute active fuel thickness
        fuel_thickness = sum(r.thickness for r in self.material_model.regions if r.is_fissionable)
        if fuel_integral > 0.0 and fuel_thickness > 0.0:
            avg_fuel_power = fuel_integral / fuel_thickness
            normalized_power = raw_power / avg_fuel_power
        else:
            normalized_power = raw_power

        return HeterogeneousSolverResult(
            keff=k_m,
            flux=final_flux,
            interior_flux=final_interior,
            normalized_power=normalized_power,
            face_currents=face_currents,
            iterations=iteration,
            converged=converged,
            history=history,
            runtime_seconds=elapsed,
            mesh=self.mesh,
            material_model=self.material_model,
            config=self.config,
        )
