"""Publication-quality visualization functions for reactor physics and numerical verification."""

from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neutron_diffusion.mesh import UniformMesh1D
from neutron_diffusion.solver import SolverResult


def set_academic_style() -> None:
    """Apply consistent academic journal formatting to matplotlib figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.grid": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_flux_comparison(
    mesh: UniformMesh1D,
    numerical_flux: np.ndarray,
    analytical_flux: np.ndarray,
    output_path: str | Path,
    title: str = "Scalar Neutron Flux Distribution (Numerical vs. Analytical)",
) -> None:
    """Generate two-panel verification figure: flux comparison (top) and pointwise error (bottom).

    Parameters
    ----------
    mesh : UniformMesh1D
        Spatial grid.
    numerical_flux : np.ndarray
        Converged numerical flux (length $N+1$).
    analytical_flux : np.ndarray
        Exact analytical reference flux (length $N+1$).
    output_path : str or Path
        Destination filepath for the saved PNG figure.
    title : str, optional
        Main figure title.
    """
    set_academic_style()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )

    x = mesh.x
    ax_top.plot(x, analytical_flux, "k-", label=r"Analytical Fundamental Mode $\phi(x) = \sqrt{2/L}\sin(\pi x / L)$")
    ax_top.plot(
        x,
        numerical_flux,
        "r--",
        dashes=(4, 3),
        label=f"Numerical FD Solution ($N={mesh.num_cells}$ cells)",
    )
    ax_top.set_ylabel(r"Normalized Scalar Flux $\phi(x)$ [$\mathrm{cm}^{-1/2}$]")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right", frameon=True)

    # Pointwise residual
    residual = numerical_flux - analytical_flux
    ax_bot.plot(x, residual, color="navy", lw=1.5, label=r"Residual $\phi_{\mathrm{num}}(x) - \phi_{\mathrm{ana}}(x)$")
    ax_bot.axhline(0.0, color="gray", linestyle=":", lw=1.0)
    ax_bot.set_xlabel("Spatial Coordinate $x$ [cm]")
    ax_bot.set_ylabel(r"Error [$\mathrm{cm}^{-1/2}$]")
    ax_bot.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_power_distribution(
    mesh: UniformMesh1D,
    power: np.ndarray,
    output_path: str | Path,
    analytical_flux_shape: np.ndarray | None = None,
) -> None:
    """Plot normalized reactor power distribution across the slab core.

    Parameters
    ----------
    mesh : UniformMesh1D
        Spatial grid.
    power : np.ndarray
        Normalized relative power distribution (average = 1.0).
    output_path : str or Path
        Destination filepath for saved figure.
    analytical_flux_shape : np.ndarray, optional
        Analytical fundamental mode normalized to average 1.0 for comparison.
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    x = mesh.x
    ax.plot(x, power, color="darkred", lw=2.2, label=r"Normalized Relative Power $P(x) / \bar{P}$")

    if analytical_flux_shape is not None:
        # Scale to average 1.0
        avg_ana = mesh.integrate(analytical_flux_shape) / mesh.geometry.length
        scaled_ana = analytical_flux_shape / avg_ana if avg_ana > 0 else analytical_flux_shape
        ax.plot(x, scaled_ana, "k--", dashes=(4, 3), label="Analytical Reference Shape")

    ax.axhline(1.0, color="gray", linestyle=":", lw=1.2, label="Core Average Power (1.0)")
    ax.set_xlabel("Core Position $x$ [cm]")
    ax.set_ylabel(r"Relative Fission Power Density $P(x) / \bar{P}$ [-]")
    ax.set_title("Normalized Reactor Power Distribution Across Homogeneous Slab")
    ax.set_xlim(0, mesh.geometry.length)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_convergence_history(
    result: SolverResult,
    output_path: str | Path,
) -> None:
    """Plot power iteration convergence history showing eigenvalue and residuals.

    Parameters
    ----------
    result : SolverResult
        Converged power iteration result with history.
    output_path : str or Path
        Destination filepath.
    """
    set_academic_style()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)

    iters = result.history.iterations
    keff = result.history.keff
    delta_k = result.history.delta_keff
    flux_res = result.history.flux_residual
    src_res = result.history.source_residual

    # Top: Eigenvalue progression
    ax_top.plot(iters, keff, color="crimson", marker="o", markersize=3, label=r"$k_{\mathrm{eff}}^{(m)}$ trajectory")
    ax_top.axhline(result.keff, color="black", linestyle="--", label=f"Converged $k_{{\\mathrm{{eff}}}} = {result.keff:.7f}$")
    ax_top.set_ylabel(r"Eigenvalue $k_{\mathrm{eff}}$ [-]")
    ax_top.set_title(f"Power Iteration Convergence ({result.iterations} iterations, tol={result.config.tolerance_k:.1e})")
    ax_top.legend(loc="lower right", frameon=True)

    # Bottom: Residuals log scale
    ax_bot.semilogy(iters, delta_k, color="darkblue", marker="s", markersize=3, label=r"Eigenvalue change $|k^{(m)} - k^{(m-1)}|$")
    ax_bot.semilogy(iters, flux_res, color="forestgreen", marker="^", markersize=3, label=r"Relative flux change $\|\phi^{(m)} - \phi^{(m-1)}\|_2$")
    ax_bot.semilogy(iters, src_res, color="darkorange", linestyle="--", label=r"Source residual $\|A\phi - \frac{1}{k}F\phi\|_2$")
    ax_bot.axhline(result.config.tolerance_k, color="gray", linestyle=":", label=f"Tolerance ($10^{{{int(np.log10(result.config.tolerance_k))}}}$)")
    ax_bot.set_xlabel("Iteration Number $m$")
    ax_bot.set_ylabel("Convergence Metric (log scale)")
    ax_bot.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_mesh_convergence(
    dx_list: list[float],
    keff_rel_errors: list[float],
    flux_l2_errors: list[float],
    output_path: str | Path,
    keff_order: float,
    flux_order: float,
) -> None:
    """Plot spatial mesh convergence on log-log axes demonstrating observed order.

    Parameters
    ----------
    dx_list : list[float]
        Mesh spacings $\\Delta x$.
    keff_rel_errors : list[float]
        Relative errors in $k_{\\text{eff}}$.
    flux_l2_errors : list[float]
        $L_2$ errors in normalized flux.
    output_path : str or Path
        Destination filepath.
    keff_order : float
        Observed convergence slope for $k_{\\text{eff}}$.
    flux_order : float
        Observed convergence slope for flux.
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    dx_arr = np.array(dx_list)
    ax.loglog(
        dx_arr,
        keff_rel_errors,
        "ro-",
        label=rf"$k_{{\mathrm{{eff}}}}$ Relative Error (Observed order $p \approx {keff_order:.2f}$)",
    )
    ax.loglog(
        dx_arr,
        flux_l2_errors,
        "bs--",
        label=r"Flux $L_2$ Error (Exact modal preservation, $< 10^{-9}$)",
    )

    # Theoretical O(dx^2) reference line
    ref_dx = np.linspace(min(dx_arr), max(dx_arr), 50)
    ref_scale = keff_rel_errors[0] / (dx_arr[0] ** 2)
    ref_line = ref_scale * (ref_dx ** 2)
    ax.loglog(ref_dx, ref_line, "k:", lw=1.5, label=r"Theoretical $O(\Delta x^2)$ Asymptotic Slope")

    ax.set_xlabel(r"Mesh Grid Spacing $\Delta x$ [cm] (log scale)")
    ax.set_ylabel("Discretization Error (log scale)")
    ax.set_title(r"Spatial Mesh Refinement Study (Second-Order FD Scheme)")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_parameter_sensitivity(
    sensitivity_results: dict[str, dict[str, list[float]]],
    baseline_keff: float,
    output_path: str | Path,
) -> None:
    """Plot 4-panel parameter sensitivity curves for $L, \\Sigma_a, D, \\nu\\Sigma_f$.

    Parameters
    ----------
    sensitivity_results : dict
        Nested dictionary with keys 'length', 'sigma_a', 'D', 'nu_sigma_f',
        each containing 'param_values' and 'keff_values'.
    baseline_keff : float
        Reference baseline $k_{\\text{eff}}$ value.
    output_path : str or Path
        Destination filepath.
    """
    set_academic_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    ax_l, ax_sa = axes[0, 0], axes[0, 1]
    ax_d, ax_nf = axes[1, 0], axes[1, 1]

    # 1. Slab Length L
    l_data = sensitivity_results["length"]
    ax_l.plot(l_data["param_values"], l_data["keff_values"], "b-o", markersize=4)
    ax_l.axhline(1.0, color="gray", linestyle=":", label="Critical ($k_{\\mathrm{eff}}=1$)")
    ax_l.set_xlabel(r"Slab Length $L$ [cm]")
    ax_l.set_ylabel(r"$k_{\mathrm{eff}}$ [-]")
    ax_l.set_title(r"Sensitivity to Core Size $L$")
    ax_l.legend(loc="lower right", frameon=True)

    # 2. Absorption cross section Sigma_a
    sa_data = sensitivity_results["sigma_a"]
    ax_sa.plot(sa_data["param_values"], sa_data["keff_values"], "r-s", markersize=4)
    ax_sa.axhline(1.0, color="gray", linestyle=":", label="Critical ($k_{\\mathrm{eff}}=1$)")
    ax_sa.set_xlabel(r"Absorption Cross Section $\Sigma_a$ [$\mathrm{cm}^{-1}$]")
    ax_sa.set_ylabel(r"$k_{\mathrm{eff}}$ [-]")
    ax_sa.set_title(r"Sensitivity to Absorption $\Sigma_a$")
    ax_sa.legend(loc="upper right", frameon=True)

    # 3. Diffusion coefficient D
    d_data = sensitivity_results["D"]
    ax_d.plot(d_data["param_values"], d_data["keff_values"], "g-^", markersize=4)
    ax_d.axhline(1.0, color="gray", linestyle=":", label="Critical ($k_{\\mathrm{eff}}=1$)")
    ax_d.set_xlabel(r"Diffusion Coefficient $D$ [cm]")
    ax_d.set_ylabel(r"$k_{\mathrm{eff}}$ [-]")
    ax_d.set_title(r"Sensitivity to Leakage / Diffusion $D$")
    ax_d.legend(loc="upper right", frameon=True)

    # 4. Neutron production cross section nuSigma_f
    nf_data = sensitivity_results["nu_sigma_f"]
    ax_nf.plot(nf_data["param_values"], nf_data["keff_values"], "m-d", markersize=4)
    ax_nf.axhline(1.0, color="gray", linestyle=":", label="Critical ($k_{\\mathrm{eff}}=1$)")
    ax_nf.set_xlabel(r"Production Cross Section $\nu\Sigma_f$ [$\mathrm{cm}^{-1}$]")
    ax_nf.set_ylabel(r"$k_{\mathrm{eff}}$ [-]")
    ax_nf.set_title(r"Sensitivity to Fission Production $\nu\Sigma_f$")
    ax_nf.legend(loc="lower right", frameon=True)

    plt.suptitle("Reactor Physics Eigenvalue Parameter Sensitivity Analysis", y=1.01, fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_flux_sensitivity(
    length_flux_profiles: list[tuple[float, np.ndarray, np.ndarray]],
    output_path: str | Path,
) -> None:
    """Plot normalized flux profiles across varying slab widths.

    Parameters
    ----------
    length_flux_profiles : list of (L, x_array, normalized_flux)
        List containing core thickness and corresponding flux profile.
    output_path : str or Path
        Destination filepath.
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    colors = ["teal", "navy", "darkorange", "purple"]
    for i, (l_val, x_arr, flx) in enumerate(length_flux_profiles):
        # normalize coordinate x/L to show buckling shape preservation
        ax.plot(
            x_arr / l_val,
            flx,
            color=colors[i % len(colors)],
            lw=2.0,
            label=f"$L = {l_val:.0f}$ cm",
        )

    ax.set_xlabel("Relative Core Position $x / L$ [-]")
    ax.set_ylabel(r"Normalized Flux $\phi(x/L)$ [$\mathrm{cm}^{-1/2}$]")
    ax.set_title("Fundamental Mode Flux Profiles Scaled Across Core Sizes")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_core_reflector_flux(
    result: Any,
    output_path: str | Path,
) -> None:
    """Plot scalar flux and fission power density for reflected core with shaded regions.

    Parameters
    ----------
    result : HeterogeneousSolverResult
        Converged solution result.
    output_path : str or Path
        Target filepath.
    """
    set_academic_style()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.5]}
    )

    x = result.mesh.x
    model = result.material_model

    # Background shading for regions
    colors = {"left_reflector": "#e0f2fe", "fuel_core": "#ffedd5", "right_reflector": "#e0f2fe"}
    for reg in model.regions:
        c = colors.get(reg.name, "#f3f4f6")
        label_reg = "Reflector" if "reflector" in reg.name else "Active Core"
        ax_top.axvspan(reg.x_min, reg.x_max, color=c, alpha=0.55)
        ax_bot.axvspan(reg.x_min, reg.x_max, color=c, alpha=0.55)

    # Vertical interface markers
    for int_x in model.interface_locations:
        ax_top.axvline(int_x, color="dimgray", linestyle="--", lw=1.2)
        ax_bot.axvline(int_x, color="dimgray", linestyle="--", lw=1.2)

    # Top: Flux
    ax_top.plot(x, result.flux, color="navy", lw=2.2, label=r"Scalar Flux $\phi(x)$")
    ax_top.set_ylabel(r"Normalized Flux $\phi(x)$ [$\mathrm{cm}^{-1/2}$]")
    ax_top.set_title(f"Heterogeneous Core + Reflector Neutron Flux ($k_{{\\mathrm{{eff}}}} = {result.keff:.6f}$)")
    ax_top.legend(loc="upper right", frameon=True)

    # Annotations
    for reg in model.regions:
        mid_x = 0.5 * (reg.x_min + reg.x_max)
        lbl = "Reflector" if "reflector" in reg.name else "Fuel Core"
        ax_top.text(
            mid_x,
            np.max(result.flux) * 0.90,
            lbl,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Bottom: Normalized power
    ax_bot.plot(x, result.normalized_power, color="darkred", lw=2.0, label=r"Power Density $P(x)/\bar{P}_{\mathrm{core}}$")
    ax_bot.axhline(1.0, color="gray", linestyle=":", label="Average Core Power (1.0)")
    ax_bot.set_xlabel("Slab Coordinate $x$ [cm]")
    ax_bot.set_ylabel(r"Relative Power [-]")
    ax_bot.set_ylim(bottom=-0.05)
    ax_bot.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_bare_vs_reflector(
    mesh_bare: Any,
    bare_flux: np.ndarray,
    mesh_refl: Any,
    refl_flux: np.ndarray,
    fuel_thickness: float,
    reflector_thickness: float,
    bare_keff: float,
    refl_keff: float,
    output_path: str | Path,
) -> None:
    """Plot overlaid flux distributions comparing bare vs. reflected cores demonstrating reflector savings."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    # Shift bare core coordinates to align with core center
    # Reflected core has fuel on [reflector_thickness, reflector_thickness + fuel_thickness]
    x_refl = mesh_refl.x
    x_bare_shifted = mesh_bare.x + reflector_thickness

    # Core boundaries
    core_left = reflector_thickness
    core_right = reflector_thickness + fuel_thickness
    ax.axvspan(core_left, core_right, color="#ffedd5", alpha=0.45, label="Active Core Region")
    ax.axvspan(0.0, core_left, color="#e0f2fe", alpha=0.45, label="Reflector Region")
    ax.axvspan(core_right, mesh_refl.geometry.length, color="#e0f2fe", alpha=0.45)

    delta_k = refl_keff - bare_keff
    ax.plot(
        x_bare_shifted,
        bare_flux,
        "k--",
        lw=2.0,
        label=rf"Bare Core ($k_{{\mathrm{{eff}}}} = {bare_keff:.5f}$)",
    )
    ax.plot(
        x_refl,
        refl_flux,
        color="crimson",
        lw=2.2,
        label=rf"Reflected Core ($k_{{\mathrm{{eff}}}} = {refl_keff:.5f}$, $\Delta k = +{delta_k:.5f}$)",
    )

    ax.axvline(core_left, color="gray", linestyle=":", lw=1.2)
    ax.axvline(core_right, color="gray", linestyle=":", lw=1.2)

    ax.set_xlabel("Slab Coordinate $x$ [cm]")
    ax.set_ylabel(r"Normalized Flux $\phi(x)$ [$\mathrm{cm}^{-1/2}$]")
    ax.set_title("Reflector Savings Effect: Bare vs. Reflected Slab Reactor")
    ax.set_xlim(0, mesh_refl.geometry.length)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_critical_fuel_thickness(
    fuel_thicknesses: list[float],
    keff_values: list[float],
    critical_thickness: float,
    output_path: str | Path,
) -> None:
    """Plot curve of effective multiplication factor vs. fuel thickness showing numerical critical dimension."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(fuel_thicknesses, keff_values, "bo-", lw=2.0, markersize=6, label=r"$k_{\mathrm{eff}}(T_{\mathrm{fuel}})$")
    ax.axhline(1.0, color="crimson", linestyle="--", lw=1.5, label=r"Critical Target ($k_{\mathrm{eff}} = 1.0$)")
    ax.axvline(
        critical_thickness,
        color="darkgreen",
        linestyle=":",
        lw=1.8,
        label=rf"Critical Thickness $T_{{\mathrm{{crit}}}} \approx {critical_thickness:.2f}$ cm",
    )

    ax.plot([critical_thickness], [1.0], "s", color="darkgreen", markersize=9, zorder=5)

    ax.set_xlabel(r"Active Fuel Thickness $T_{\mathrm{fuel}}$ [cm]")
    ax.set_ylabel(r"Effective Multiplication Factor $k_{\mathrm{eff}}$ [-]")
    ax.set_title("Numerical Critical Fuel Thickness Determination (Fixed 20 cm Reflector)")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_reflector_sensitivity(
    results_dict: dict[str, dict[str, list[float]]],
    output_path: str | Path,
) -> None:
    """Plot 4-panel sensitivity analysis for heterogeneous parameters."""
    set_academic_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    ax_rt, ax_rsa = axes[0, 0], axes[0, 1]
    ax_fsa, ax_fnsf = axes[1, 0], axes[1, 1]

    # 1. Reflector thickness
    rt_data = results_dict["reflector_thickness"]
    ax_rt.plot(rt_data["param_values"], rt_data["keff_values"], "b-o", markersize=5)
    ax_rt.axhline(1.0, color="gray", linestyle=":")
    ax_rt.set_xlabel(r"Reflector Thickness $T_{\mathrm{refl}}$ [cm]")
    ax_rt.set_ylabel(r"$k_{\mathrm{eff}}$ [-]")
    ax_rt.set_title(r"Reflector Thickness Effect")

    # 2. Reflector Sigma_a
    rsa_data = results_dict["reflector_sigma_a"]
    ax_rsa.plot(rsa_data["param_values"], rsa_data["keff_values"], "r-s", markersize=5)
    ax_rsa.axhline(1.0, color="gray", linestyle=":")
    ax_rsa.set_xlabel(r"Reflector Absorption $\Sigma_{a,\mathrm{refl}}$ [$\mathrm{cm}^{-1}$]")
    ax_rsa.set_ylabel(r"$k_{\mathrm{eff}}$ [-]")
    ax_rsa.set_title(r"Reflector Absorption Sensitivity")

    # 3. Fuel Sigma_a
    fsa_data = results_dict["fuel_sigma_a"]
    ax_fsa.plot(fsa_data["param_values"], fsa_data["keff_values"], "g-^", markersize=5)
    ax_fsa.axhline(1.0, color="gray", linestyle=":")
    ax_fsa.set_xlabel(r"Fuel Absorption $\Sigma_{a,\mathrm{fuel}}$ [$\mathrm{cm}^{-1}$]")
    ax_fsa.set_ylabel(r"$k_{\mathrm{eff}}$ [-]")
    ax_fsa.set_title(r"Fuel Absorption Sensitivity")

    # 4. Fuel nuSigma_f
    fnsf_data = results_dict["fuel_nu_sigma_f"]
    ax_fnsf.plot(fnsf_data["param_values"], fnsf_data["keff_values"], "m-d", markersize=5)
    ax_fnsf.axhline(1.0, color="gray", linestyle=":")
    ax_fnsf.set_xlabel(r"Fuel Production $\nu\Sigma_{f,\mathrm{fuel}}$ [$\mathrm{cm}^{-1}$]")
    ax_fnsf.set_ylabel(r"$k_{\mathrm{eff}}$ [-]")
    ax_fnsf.set_title(r"Fuel Fission Production Sensitivity")

    plt.suptitle("Heterogeneous Core + Reflector Parameter Sensitivities", y=1.01, fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_heterogeneous_material_map(
    model: Any,
    mesh: Any,
    output_path: str | Path,
) -> None:
    """Plot step-function spatial maps of D(x), Sigma_a(x), and nuSigma_f(x)."""
    set_academic_style()
    fig, (ax_d, ax_sa, ax_nsf) = plt.subplots(3, 1, figsize=(7.5, 7.0), sharex=True)

    d_nodal, sa_nodal, nsf_nodal, _ = model.sample_nodal_properties(mesh)
    x = mesh.x

    # D(x)
    ax_d.plot(x, d_nodal, color="teal", lw=2.0)
    ax_d.set_ylabel(r"$D(x)$ [cm]")
    ax_d.set_title("Piecewise Material Cross-Section Profiles")

    # Sigma_a(x)
    ax_sa.plot(x, sa_nodal, color="crimson", lw=2.0)
    ax_sa.set_ylabel(r"$\Sigma_a(x)$ [$\mathrm{cm}^{-1}$]")

    # nuSigma_f(x)
    ax_nsf.plot(x, nsf_nodal, color="purple", lw=2.0)
    ax_nsf.set_ylabel(r"$\nu\Sigma_f(x)$ [$\mathrm{cm}^{-1}$]")
    ax_nsf.set_xlabel("Spatial Coordinate $x$ [cm]")

    for ax in (ax_d, ax_sa, ax_nsf):
        for int_x in model.interface_locations:
            ax.axvline(int_x, color="gray", linestyle="--", lw=1.0)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_heterogeneous_convergence(
    dx_list: list[float],
    error_list: list[float],
    output_path: str | Path,
) -> None:
    """Plot spatial convergence for the heterogeneous core-reflector configuration."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    dx_arr = np.array(dx_list)
    err_arr = np.array(error_list)

    ax.loglog(dx_arr, err_arr, "ro-", lw=1.8, label=r"Reflected Core $k_{\mathrm{eff}}$ Discretization Error")

    # Reference O(dx^2) slope
    ref_scale = err_arr[0] / (dx_arr[0] ** 2)
    ref_line = ref_scale * (dx_arr ** 2)
    ax.loglog(dx_arr, ref_line, "k:", lw=1.5, label=r"Theoretical $O(\Delta x^2)$ Asymptotic Slope")

    ax.set_xlabel(r"Mesh Grid Spacing $\Delta x$ [cm] (log scale)")
    ax.set_ylabel("Discretization Error (log scale)")
    ax.set_title("Spatial Mesh Convergence on Heterogeneous Interface Problem")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
