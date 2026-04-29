"""
run.py
Runner file to control all possible runs and plots for:
– Noiseless simulation
– Fake hardware simulation
– Real hardware run

NOTE: IBM API token must be installed locally to run the real hardware part of this script
run this in terminal to save your IBM API token (replace YOUR_IBM_API_KEY with the actual key):
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='YOUR_IBM_API_KEY', set_as_default=True, overwrite=True)"
check that it saved:
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; print(QiskitRuntimeService.saved_accounts())" 
"""

import argparse
from datetime import datetime
from ewfs.analysis.lf_violations import LF_violation
from ewfs.circuits.accuracy_test_circuits import ACCURACY_TEST_BUILDERS
from ewfs.circuits.agents import AGENTS
from ewfs.experiments.fake_hardware import prepare_fake_hardware_run, run_fake_hardware_for_backend
from ewfs.experiments.noiseless_simulation import run_noiseless_simulation
from ewfs.experiments.real_hardware import (
    prepare_real_hardware_run,
    run_grouped_real_hardware_for_backend,
    run_real_hardware_for_backend,
)
from ewfs.paths import PROJECT_ROOT


# -----------------------------------------------------------------------------
# RUN CONFIGURATION

# Noiseless simulation
DO_NOISELESS_SIM = True
NOISELESS_SHOTS = 100_000
SAVE_NOISELESS_DATA = True

# Plot noiseless circuits together with noiseless simulation
PLOT_NOISELESS_CIRCUITS = True

# IBM transpilation plots
DO_IBM_TRANSPILATION = True
SAVE_IBM_TRANSPILATION_PLOTS = True

# Fake hardware noise simulation
DO_FAKE_HARDWARE_SIM = True
FAKE_HARDWARE_SHOTS = 10_000

# Real hardware
DO_REAL_HARDWARE = True
REAL_HARDWARE_SHOTS = 1_000

# Accuracy-test circuits
INCLUDE_ACCURACY_TEST_CIRCUITS = True
NOISELESS_ACCURACY_TEST_SHOTS = NOISELESS_SHOTS
FAKE_HARDWARE_ACCURACY_TEST_SHOTS = FAKE_HARDWARE_SHOTS
REAL_HARDWARE_ACCURACY_TEST_SHOTS = 1000

# Backends to use
REAL_BACKENDS = {
    "ibm_torino": False,
    "ibm_kingston": False,
    "ibm_fez": False,
    "ibm_marrakesh": True,
}

# LF violation calculation
CALCULATE_LF_VIOLATIONS = True
LF_AGENTS = ["Betting Agent", "Guessing Agent", "Reflex Agent", "Always 3/4 Agent"]

# -----------------------------------------------------------------------------
# Helpers for LF violation calculation from saved data files

def get_latest_noiseless_file():
    """Return the newest noiseless simulation JSON file."""
    data_dir = PROJECT_ROOT / "data" / "data_noiseless_simulation"
    candidates = sorted(data_dir.glob("noiseless_simulation_*/noiseless_simulation.json"))
    return candidates[-1] if candidates else None


def get_latest_fake_file(backend_name: str):
    """Return the newest fake hardware JSON file for one backend."""
    data_dir = PROJECT_ROOT / "data" / "data_fake_hardware"
    candidates = sorted(data_dir.glob(f"{backend_name}_*/fake_hardware_noise_sim.json"))
    return candidates[-1] if candidates else None


def get_latest_real_file(backend_name: str):
    """Return the newest real hardware JSON file for one backend."""
    data_dir = PROJECT_ROOT / "data" / "data_real_hardware"
    candidates = sorted(data_dir.glob(f"{backend_name}_*/real_hardware_run.json"))
    return candidates[-1] if candidates else None


def print_lf_violations(label, data_path):
    """Print LF violations for all configured agents from one saved result file."""
    if data_path is None:
        print(f"\n{label} LF violations could not be calculated: no data file found.")
        return

    print(f"\n=== LF violations: {label} ===")
    for agent in LF_AGENTS:
        print(f"  {agent}: S = {LF_violation(str(data_path), agent=agent)}")

# -----------------------------------------------------------------------------
# IBM Quantum Platform helpers

def get_runtime_service():
    """Connect to IBM Quantum Platform. Credentials must be available locally."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService()


def get_real_backend(service, backend_name: str):
    """Get a real backend handle."""
    return service.backend(backend_name)


# -----------------------------------------------------------------------------
# Main runner:

def parse_args():
    """Parse runner CLI overrides."""
    # These options are only used when this file is started from the terminal.
    # They let us temporarily override a few settings without editing the
    # configuration constants above.
    parser = argparse.ArgumentParser(description="Run EWFS experiments and optional accuracy-test circuits.")
    parser.add_argument(
        "--include-accuracy-tests",
        dest="include_accuracy_tests",
        action="store_true",
        help="Include the hard-coded accuracy-test circuits alongside the main EWFS circuits.",
    )
    parser.add_argument(
        "--exclude-accuracy-tests",
        dest="include_accuracy_tests",
        action="store_false",
        help="Run only the main EWFS circuits.",
    )
    parser.add_argument(
        "--shots-main",
        type=int,
        default=None,
        help="Override the main-circuit shot count for all enabled run modes.",
    )
    parser.add_argument(
        "--shots-accuracy-tests",
        type=int,
        default=None,
        help="Override the accuracy-test circuit shot count for all enabled run modes.",
    )
    # If neither --include-accuracy-tests nor --exclude-accuracy-tests is given,
    # use the value from INCLUDE_ACCURACY_TEST_CIRCUITS above.
    parser.set_defaults(include_accuracy_tests=INCLUDE_ACCURACY_TEST_CIRCUITS)
    return parser.parse_args()


def run_all(
    include_accuracy_tests=INCLUDE_ACCURACY_TEST_CIRCUITS,
    shots_main=None,
    shots_accuracy_tests=None,
):
    """Run whichever parts are enabled above."""
    main_builders = AGENTS
    accuracy_test_builders = ACCURACY_TEST_BUILDERS

    noiseless_main_shots = NOISELESS_SHOTS if shots_main is None else shots_main
    fake_main_shots = FAKE_HARDWARE_SHOTS if shots_main is None else shots_main
    real_main_shots = REAL_HARDWARE_SHOTS if shots_main is None else shots_main

    noiseless_accuracy_test_shots = (
        NOISELESS_ACCURACY_TEST_SHOTS if shots_accuracy_tests is None else shots_accuracy_tests
    )
    fake_accuracy_test_shots = (
        FAKE_HARDWARE_ACCURACY_TEST_SHOTS if shots_accuracy_tests is None else shots_accuracy_tests
    )
    real_accuracy_test_shots = (
        REAL_HARDWARE_ACCURACY_TEST_SHOTS if shots_accuracy_tests is None else shots_accuracy_tests
    )

    if DO_NOISELESS_SIM:
        noiseless_folder_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_noiseless_simulation(
            shots=noiseless_main_shots,
            save=SAVE_NOISELESS_DATA,
            make_plots=PLOT_NOISELESS_CIRCUITS,
            agent_builders=main_builders,
            folder_ts=noiseless_folder_ts,
            plots_subdir="circuit_plots",
        )
        if CALCULATE_LF_VIOLATIONS and SAVE_NOISELESS_DATA:
            print_lf_violations("Noiseless simulation", get_latest_noiseless_file())
        if include_accuracy_tests:
            run_noiseless_simulation(
                shots=noiseless_accuracy_test_shots,
                save=SAVE_NOISELESS_DATA,
                make_plots=PLOT_NOISELESS_CIRCUITS,
                agent_builders=accuracy_test_builders,
                folder_ts=noiseless_folder_ts,
                result_filename="accuracy_test_noiseless_simulation.json",
                plots_subdir="accuracy_test_circuit_plots",
            )

    needs_ibm_backend = DO_IBM_TRANSPILATION or DO_FAKE_HARDWARE_SIM or DO_REAL_HARDWARE
    if not needs_ibm_backend:
        return

    print("\n=== IBM Quantum Platform backend ===")
    service = get_runtime_service()

    for backend_name, enabled in REAL_BACKENDS.items():
        if not enabled:
            continue

        backend = get_real_backend(service, backend_name)
        print(f"\n=== Backend: {backend.name} ===")

        fake_transpiled_by_agent = None
        fake_folder_ts = None
        real_transpiled_by_agent = None
        real_folder_ts = None

        if DO_FAKE_HARDWARE_SIM:
            fake_transpiled_by_agent, fake_folder_ts = prepare_fake_hardware_run(
                backend=backend,
                save_plots=SAVE_IBM_TRANSPILATION_PLOTS,
                agent_builders=main_builders,
                plots_subdir="transpiled_agents",
            )

        if DO_REAL_HARDWARE:
            real_transpiled_by_agent, real_folder_ts = prepare_real_hardware_run(
                backend=backend,
                save_plots=SAVE_IBM_TRANSPILATION_PLOTS,
                agent_builders=main_builders,
                plots_subdir="transpiled_agents",
            )

        if DO_IBM_TRANSPILATION and not DO_FAKE_HARDWARE_SIM and not DO_REAL_HARDWARE:
            _, transpilation_only_folder_ts = prepare_real_hardware_run(
                backend=backend,
                save_plots=SAVE_IBM_TRANSPILATION_PLOTS,
                agent_builders=main_builders,
                plots_subdir="transpiled_agents",
            )
            if include_accuracy_tests:
                prepare_real_hardware_run(
                    backend=backend,
                    save_plots=SAVE_IBM_TRANSPILATION_PLOTS,
                    folder_ts=transpilation_only_folder_ts,
                    agent_builders=accuracy_test_builders,
                    plots_subdir="accuracy_test_transpiled_circuits",
                )

        if DO_FAKE_HARDWARE_SIM:
            run_fake_hardware_for_backend(
                backend=backend,
                transpiled_by_agent=fake_transpiled_by_agent,
                shots=fake_main_shots,
                folder_ts=fake_folder_ts,
            )
            if CALCULATE_LF_VIOLATIONS:
                print_lf_violations(
                    f"Fake hardware simulation ({backend.name})",
                    get_latest_fake_file(backend.name),
                )
            if include_accuracy_tests:
                fake_accuracy_test_transpiled, fake_accuracy_test_folder_ts = prepare_fake_hardware_run(
                    backend=backend,
                    save_plots=SAVE_IBM_TRANSPILATION_PLOTS,
                    folder_ts=fake_folder_ts,
                    agent_builders=accuracy_test_builders,
                    plots_subdir="accuracy_test_transpiled_circuits",
                )
                run_fake_hardware_for_backend(
                    backend=backend,
                    transpiled_by_agent=fake_accuracy_test_transpiled,
                    shots=fake_accuracy_test_shots,
                    folder_ts=fake_accuracy_test_folder_ts,
                    result_filename="accuracy_test_fake_hardware_noise_sim.json",
                )

        if DO_REAL_HARDWARE:
            if include_accuracy_tests:
                real_accuracy_test_transpiled, real_accuracy_test_folder_ts = prepare_real_hardware_run(
                    backend=backend,
                    save_plots=SAVE_IBM_TRANSPILATION_PLOTS,
                    folder_ts=real_folder_ts,
                    agent_builders=accuracy_test_builders,
                    plots_subdir="accuracy_test_transpiled_circuits",
                )
                real_run_dirs = run_grouped_real_hardware_for_backend(
                    backend=backend,
                    folder_ts=real_accuracy_test_folder_ts,
                    job_groups=[
                        {
                            "group_key": "main",
                            "transpiled_by_agent": real_transpiled_by_agent,
                            "shots": real_main_shots,
                            "result_filename": "real_hardware_run.json",
                            "timing_filename": "scheduler_timing_metadata.json",
                            "job_info_filename": "job_info.json",
                            "raw_result_filename": "raw_sampler_result.pkl",
                        },
                        {
                            "group_key": "accuracy_tests",
                            "transpiled_by_agent": real_accuracy_test_transpiled,
                            "shots": real_accuracy_test_shots,
                            "result_filename": "accuracy_test_real_hardware_run.json",
                            "timing_filename": "accuracy_test_scheduler_timing_metadata.json",
                            "job_info_filename": "accuracy_test_job_info.json",
                            "raw_result_filename": "accuracy_test_raw_sampler_result.pkl",
                        },
                    ],
                )
                real_run_dir = real_run_dirs["main"]
            else:
                real_run_dir = run_real_hardware_for_backend(
                    backend=backend,
                    transpiled_by_agent=real_transpiled_by_agent,
                    shots=real_main_shots,
                    folder_ts=real_folder_ts,
                )

            if CALCULATE_LF_VIOLATIONS:
                print_lf_violations(
                    f"Real hardware run ({backend.name})",
                    get_latest_real_file(backend.name),
                )


def main():
    # Read optional terminal arguments, then pass them into the main runner.
    # Example: python scripts/run_experiment.py --shots-main 5000 --exclude-accuracy-tests
    args = parse_args()
    run_all(
        include_accuracy_tests=args.include_accuracy_tests,
        shots_main=args.shots_main,
        shots_accuracy_tests=args.shots_accuracy_tests,
    )


if __name__ == "__main__":
    main()
