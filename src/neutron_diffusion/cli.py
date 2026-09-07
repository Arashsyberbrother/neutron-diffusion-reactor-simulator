"""Command-line interface (CLI) for executing simulations and parameter studies."""

import argparse
import sys
from pathlib import Path

from neutron_diffusion.config import SimulationConfig, get_baseline_config
from neutron_diffusion.experiments import (
    run_baseline_experiment,
    run_mesh_convergence_study,
    run_parameter_study,
)
from neutron_diffusion.reflector_study import (
    compare_bare_vs_reflected,
    run_core_reflector_baseline,
    run_heterogeneous_convergence_study,
    run_heterogeneous_parameter_study,
    study_critical_fuel_thickness,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser with scientific subcommands."""
    parser = argparse.ArgumentParser(
        prog="neutron-diffusion",
        description="Scientific 1D steady-state neutron diffusion eigenvalue solver and analysis suite.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Simulation subcommand to execute.")

    # 1. Baseline (Version 1 Homogeneous)
    p_base = subparsers.add_parser("baseline", help="Run the baseline homogeneous verification experiment.")
    p_base.add_argument("--length", type=float, default=100.0, help="Core slab thickness L in cm (default: 100.0).")
    p_base.add_argument("--cells", type=int, default=100, help="Number of spatial cells N (default: 100).")
    p_base.add_argument("--D", type=float, default=1.0, help="Diffusion coefficient D in cm (default: 1.0).")
    p_base.add_argument("--sigma-a", type=float, default=0.02, help="Absorption cross section Sigma_a in cm^-1 (default: 0.02).")
    p_base.add_argument("--nu-sigma-f", type=float, default=0.025, help="Production cross section nuSigma_f in cm^-1 (default: 0.025).")
    p_base.add_argument("--tol-k", type=float, default=1.0e-7, help="Eigenvalue convergence tolerance (default: 1.0e-7).")
    p_base.add_argument("--output-dir", type=str, default="results", help="Directory to save figures and metrics.")

    # 2. Verify
    p_verify = subparsers.add_parser("verify", help="Run baseline verification and report analytical comparison.")
    p_verify.add_argument("--output-dir", type=str, default="results", help="Directory for output.")

    # 3. Convergence
    p_conv = subparsers.add_parser("convergence", help="Run spatial mesh refinement convergence study.")
    p_conv.add_argument("--cells", nargs="+", type=int, default=[20, 40, 80, 160, 320, 640], help="List of mesh cell counts.")
    p_conv.add_argument("--output-dir", type=str, default="results", help="Directory for output.")

    # 4. Sensitivity
    p_sens = subparsers.add_parser("sensitivity", help="Run reactor physics parameter sensitivity study.")
    p_sens.add_argument("--output-dir", type=str, default="results", help="Directory for output.")

    # 5. Version 2 Heterogeneous Core + Reflector Baseline
    p_het = subparsers.add_parser("heterogeneous", help="Run Version 2 core + reflector heterogeneous simulation.")
    p_het.add_argument("--output-dir", type=str, default="results", help="Directory for output.")

    # 6. Version 2 Reflector Savings Study
    p_refl = subparsers.add_parser("reflector", help="Run controlled bare vs. reflected core comparison.")
    p_refl.add_argument("--output-dir", type=str, default="results", help="Directory for output.")

    # 7. Version 2 Critical Fuel Thickness Study
    p_crit = subparsers.add_parser("critical-thickness", help="Run critical fuel core dimension sweep.")
    p_crit.add_argument("--output-dir", type=str, default="results", help="Directory for output.")

    # 8. All (runs full V1 and V2 suite)
    p_all = subparsers.add_parser("all", help="Run complete scientific pipeline (homogeneous V1 and heterogeneous V2).")
    p_all.add_argument("--output-dir", type=str, default="results", help="Directory for output.")

    return parser


def print_banner() -> None:
    """Print clean scientific header banner."""
    print("=" * 72)
    print(" 1D STEADY-STATE NEUTRON DIFFUSION REACTOR SIMULATOR")
    print(" Computational Nuclear Engineering Prototype (Version 2.0)")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for neutron-diffusion-reactor-simulator."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    print_banner()

    out_dir = Path(getattr(args, "output_dir", "results"))

    try:
        if args.command in ("baseline", "verify"):
            print(f"\n[*] Executing Baseline Homogeneous Case (Output: {out_dir})...")
            cfg = SimulationConfig(
                length=getattr(args, "length", 100.0),
                D=getattr(args, "D", 1.0),
                sigma_a=getattr(args, "sigma_a", 0.02),
                nu_sigma_f=getattr(args, "nu_sigma_f", 0.025),
                num_cells=getattr(args, "cells", 100),
                tolerance_k=getattr(args, "tol_k", 1.0e-7),
            )
            summary = run_baseline_experiment(config=cfg, output_dir=out_dir)

            res = summary["results"]
            prms = summary["parameters"]
            conv = summary["convergence"]

            print("\n-------------------------------------------------------------")
            print(" BASELINE HOMOGENEOUS SIMULATION SUMMARY")
            print("-------------------------------------------------------------")
            print(f" Core Thickness L            : {prms['length_cm']:.2f} cm")
            print(f" Diffusion Coefficient D     : {prms['diffusion_coefficient_cm']:.4f} cm")
            print(f" Absorption Cross Section Sa : {prms['absorption_cross_section_cm_inv']:.4f} cm^-1")
            print(f" Fission Production nuSf     : {prms['fission_production_cross_section_cm_inv']:.4f} cm^-1")
            print(f" Geometric Buckling Bg^2     : {prms['geometric_buckling_cm_inv_sq']:.8e} cm^-2")
            print(f" Mesh Cells (N)              : {summary['discretization']['num_cells']} (dx = {summary['discretization']['dx_cm']:.4f} cm)")
            print(f" Iterations to Convergence   : {conv['iterations']} (runtime: {conv['runtime_seconds']:.4f} s)")
            print(f" Numerical k_eff             : {res['numerical_keff']:.8f}")
            print(f" Analytical k_eff            : {res['analytical_keff']:.8f}")
            print(f" Absolute Error              : {res['absolute_error']:.8e}")
            print(f" Relative Error              : {res['relative_error']:.8e} ({res['pcm_error']:.2f} pcm)")
            print(f" Flux L2 Relative Error      : {res['flux_l2_relative_error']:.8e}")
            print("-------------------------------------------------------------")
            print(f"[+] Figures saved to: {out_dir / 'figures'}")
            print(f"[+] Metrics saved to: {out_dir / 'metrics'}")

        elif args.command == "convergence":
            print(f"\n[*] Running Spatial Mesh Convergence Study across {args.cells} cells...")
            conv_data = run_mesh_convergence_study(mesh_cells_list=args.cells, output_dir=out_dir)

            obs = conv_data["observed_convergence"]
            print("\n-------------------------------------------------------------")
            print(" MESH REFINEMENT STUDY RESULTS")
            print("-------------------------------------------------------------")
            print(f"{'Cells':>8} {'dx [cm]':>12} {'k_eff (num)':>14} {'Rel Err (k)':>14} {'L2 Err (Flux)':>14}")
            print("-" * 65)
            for r in conv_data["mesh_refinements"]:
                print(
                    f"{r['num_cells']:8d} {r['dx_cm']:12.4e} {r['numerical_keff']:14.8f} "
                    f"{r['relative_keff_error']:14.4e} {r['flux_l2_relative_error']:14.4e}"
                )
            print("-" * 65)
            print(f" Observed k_eff Convergence Order : p = {obs['keff_convergence_order']:.3f} (R^2 = {obs['keff_r_squared']:.4f})")
            print(f" Observed Flux Convergence Order  : p = {obs['flux_convergence_order']:.3f} (R^2 = {obs['flux_r_squared']:.4f})")
            print(f" Expected Theoretical FD Order    : p = {obs['expected_theoretical_order']:.1f}")
            print("-------------------------------------------------------------")

        elif args.command == "sensitivity":
            print("\n[*] Running Homogeneous Parameter Sensitivity Study...")
            run_parameter_study(output_dir=out_dir)
            print("[+] Parameter study complete. Figures and tables generated.")

        elif args.command == "heterogeneous":
            print(f"\n[*] Executing Heterogeneous Core + Reflector Simulation...")
            summary = run_core_reflector_baseline(output_dir=out_dir)
            res = summary["results"]
            geom = summary["geometry"]
            print("\n-------------------------------------------------------------")
            print(" HETEROGENEOUS CORE + REFLECTOR SUMMARY")
            print("-------------------------------------------------------------")
            print(f" Total Slab Thickness        : {geom['total_length_cm']:.2f} cm")
            print(f" Active Fuel Thickness       : {geom['fuel_thickness_cm']:.2f} cm")
            print(f" Reflector Thickness         : {geom['reflector_thickness_cm']:.2f} cm (per side)")
            print(f" Converged k_eff             : {res['keff']:.8f}")
            print(f" Outer Iterations            : {res['iterations']} ({res['runtime_seconds']:.4f} s)")
            print(f" Peak-to-Average Core Power  : {res['peak_to_average_core_power']:.4f}")
            print("-------------------------------------------------------------")

        elif args.command == "reflector":
            print(f"\n[*] Executing Controlled Bare vs. Reflected Core Study...")
            comp = compare_bare_vs_reflected(output_dir=out_dir)
            print("\n-------------------------------------------------------------")
            print(" REFLECTOR SAVING EFFECT")
            print("-------------------------------------------------------------")
            print(f" Fuel Slab Thickness         : {comp['fuel_thickness_cm']:.2f} cm")
            print(f" Bare Core k_eff             : {comp['bare_keff']:.8f}")
            print(f" Reflected Core k_eff        : {comp['reflected_keff']:.8f}")
            print(f" Reactivity Gain (Delta-k)   : +{comp['reflector_saving_delta_keff']:.8f} (+{comp['reflector_saving_pcm']:.1f} pcm)")
            print("-------------------------------------------------------------")

        elif args.command == "critical-thickness":
            print(f"\n[*] Determining Numerical Critical Fuel Thickness...")
            crit_data = study_critical_fuel_thickness(output_dir=out_dir)
            print("\n-------------------------------------------------------------")
            print(" CRITICAL FUEL THICKNESS RESULT")
            print("-------------------------------------------------------------")
            print(f" Reflector Thickness         : {crit_data['reflector_thickness_cm']:.2f} cm")
            print(f" Critical Fuel Thickness     : {crit_data['numerical_critical_fuel_thickness_cm']:.2f} cm")
            print(f" Verification k_eff          : {crit_data['critical_keff_verification']:.8f}")
            print(f" Error from Critical Target  : {crit_data['error_from_critical_pcm']:.2f} pcm")
            print("-------------------------------------------------------------")

        elif args.command == "all":
            print("\n[*] Executing Full Scientific Workflow (Version 1 & Version 2)...")
            # Version 1 Homogeneous
            run_baseline_experiment(output_dir=out_dir)
            run_mesh_convergence_study(output_dir=out_dir)
            run_parameter_study(output_dir=out_dir)
            # Version 2 Heterogeneous
            run_core_reflector_baseline(output_dir=out_dir)
            compare_bare_vs_reflected(output_dir=out_dir)
            study_critical_fuel_thickness(output_dir=out_dir)
            run_heterogeneous_parameter_study(output_dir=out_dir)
            run_heterogeneous_convergence_study(output_dir=out_dir)
            print("\n[+] Complete scientific workflow successfully finished.")
            print(f"[+] All artifacts generated under: {out_dir.resolve()}")

        return 0

    except Exception as e:
        print(f"\n[!] ERROR encountered during execution: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
