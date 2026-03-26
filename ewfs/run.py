"""
run.py
Runner file to control all possible runs and plots for:
– Noiseless simulation
– Fake hardware simulation
– Real hardware run
"""

try:
    from .noiseless_simulation import run_noiseless_simulation
    from .fake_hardware import run_fake_hardware_for_backend, prepare_fake_hardware_run
    from .real_hardware import run_real_hardware_for_backend, prepare_real_hardware_run
    from .lf_violations import LF_violation
    from .time_ordering_hardware import save_visualizations_for_run as run_time_ordering_hardware
except ImportError:
    from noiseless_simulation import run_noiseless_simulation
    from fake_hardware import run_fake_hardware_for_backend, prepare_fake_hardware_run
    from real_hardware import run_real_hardware_for_backend, prepare_real_hardware_run
    from lf_violations import LF_violation
    from time_ordering_hardware import save_visualizations_for_run as run_time_ordering_hardware
from pathlib import Path


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
FAKE_HARDWARE_SHOTS = 5_000

# Real hardware
DO_REAL_HARDWARE = True
REAL_HARDWARE_SHOTS = 5_000

# Scheduler timing / time ordering analysis for the real hardware run
DO_TIME_ORDERING_HARDWARE = True

# Backends to use
REAL_BACKENDS = {
    "ibm_torino": True,
    "ibm_kingston": False,
    "ibm_fez": False,
    "ibm_marrakesh": False,
}

# LF violation calculation
CALCULATE_LF_VIOLATIONS = True
LF_AGENTS = ["Betting Agent", "Guessing Agent", "Reflex Agent", "Always 3/4 Agent"]


def get_latest_noiseless_file():
    """Return the newest noiseless simulation JSON file."""
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "data_noiseless_simulation"
    candidates = sorted(data_dir.glob("noiseless_simulation_*/noiseless_simulation.json"))
    return candidates[-1] if candidates else None


def get_latest_fake_file(backend_name: str):
    """Return the newest fake hardware JSON file for one backend."""
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "data_fake_hardware"
    candidates = sorted(data_dir.glob(f"{backend_name}_*/fake_hardware_noise_sim.json"))
    return candidates[-1] if candidates else None


def get_latest_real_file(backend_name: str):
    """Return the newest real hardware JSON file for one backend."""
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "data_real_hardware"
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

def run_all():
    """Run whichever parts are enabled above."""
    if DO_NOISELESS_SIM:
        run_noiseless_simulation(
            shots=NOISELESS_SHOTS,
            save=SAVE_NOISELESS_DATA,
            make_plots=PLOT_NOISELESS_CIRCUITS,
        )
        if CALCULATE_LF_VIOLATIONS and SAVE_NOISELESS_DATA:
            print_lf_violations("Noiseless simulation", get_latest_noiseless_file())

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
            )

        if DO_REAL_HARDWARE:
            real_transpiled_by_agent, real_folder_ts = prepare_real_hardware_run(
                backend=backend,
                save_plots=SAVE_IBM_TRANSPILATION_PLOTS,
            )

        if DO_IBM_TRANSPILATION and not DO_FAKE_HARDWARE_SIM and not DO_REAL_HARDWARE:
            prepare_real_hardware_run(
                backend=backend,
                save_plots=SAVE_IBM_TRANSPILATION_PLOTS,
            )

        if DO_FAKE_HARDWARE_SIM:
            run_fake_hardware_for_backend(
                backend=backend,
                transpiled_by_agent=fake_transpiled_by_agent,
                shots=FAKE_HARDWARE_SHOTS,
                folder_ts=fake_folder_ts,
            )
            if CALCULATE_LF_VIOLATIONS:
                print_lf_violations(
                    f"Fake hardware simulation ({backend.name})",
                    get_latest_fake_file(backend.name),
                )

        if DO_REAL_HARDWARE:
            real_run_dir = run_real_hardware_for_backend(
                backend=backend,
                transpiled_by_agent=real_transpiled_by_agent,
                shots=REAL_HARDWARE_SHOTS,
                folder_ts=real_folder_ts,
            )

            if DO_TIME_ORDERING_HARDWARE:
                print(f"\n=== Scheduler timing / time ordering ({backend.name}) ===")
                run_time_ordering_hardware(real_run_dir)

            if CALCULATE_LF_VIOLATIONS:
                print_lf_violations(
                    f"Real hardware run ({backend.name})",
                    get_latest_real_file(backend.name),
                )


if __name__ == "__main__":
    run_all()
