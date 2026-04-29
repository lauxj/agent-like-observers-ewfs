"""
Run circuits on real IBM quantum hardware.

The script transpiles each agent circuit for a selected backend, submits the
circuits through SamplerV2, and saves the returned counts and job metadata.

NOTE: IBM API token must be installed locally to run this script
run this in terminal to save your IBM API token (replace YOUR_IBM_API_KEY with the actual key):
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='YOUR_IBM_API_KEY', set_as_default=True, overwrite=True)"
check that it saved: 
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; print(QiskitRuntimeService.saved_accounts())"
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from ewfs.experiments.ibm_transpilation import PLOT_DIR as TRANSPILATION_PLOT_DIR
from ewfs.experiments.ibm_transpilation import transpile_all_agents
from ewfs.paths import PROJECT_ROOT

# define directories
DATA_DIR_REAL = PROJECT_ROOT / "data" / "data_real_hardware"
DEFAULT_RESULT_FILENAME = "real_hardware_run.json"
DEFAULT_TIMING_FILENAME = "scheduler_timing_metadata.json"
DEFAULT_JOB_INFO_FILENAME = "job_info.json"
DEFAULT_RAW_RESULT_FILENAME = "raw_sampler_result.pkl"


# prepare for real hardware by transpiling all circuits
def prepare_real_hardware_run(
    backend,
    save_plots=True,
    folder_ts=None,
    agent_builders=None,
    plots_subdir="transpiled_agents",
):
    """Transpile all selected agent circuits before submitting a hardware job."""
    folder_ts, run_folder_name = make_run_folder_name(backend, folder_ts)
    plots_dir = None
    if save_plots:
        plots_dir = TRANSPILATION_PLOT_DIR / "real_hardware" / run_folder_name / plots_subdir

    transpiled_by_agent = transpile_all_agents(
        backend,
        save_plots=save_plots,
        plots_dir=plots_dir,
        agent_builders=agent_builders,
    )
    return transpiled_by_agent, folder_ts


# submit one hardware job and save one result file
def run_real_hardware_for_backend(
    backend,
    transpiled_by_agent,
    shots=300,
    folder_ts=None,
    result_filename=DEFAULT_RESULT_FILENAME,
    timing_filename=DEFAULT_TIMING_FILENAME,
    job_info_filename=DEFAULT_JOB_INFO_FILENAME,
    raw_result_filename=DEFAULT_RAW_RESULT_FILENAME,
):
    """Run all transpiled agent circuits on one real backend and save results."""
    print("\n=== Real hardware run ===")
    print(f"Backend: {backend.name}")
    print(f"Shots: {shots}")

    job, results, meta_info = submit_hardware_job(
        transpiled_by_agent=transpiled_by_agent,
        backend=backend,
        shots=shots,
    )

    return save_hardware_results(
        job=job,
        results=results,
        meta_info=meta_info,
        backend=backend,
        shots=shots,
        folder_ts=folder_ts,
        result_filename=result_filename,
        timing_filename=timing_filename,
        job_info_filename=job_info_filename,
        raw_result_filename=raw_result_filename,
    )


# submit one combined hardware job and split the returned data into result files
def run_grouped_real_hardware_for_backend(
    backend,
    job_groups,
    folder_ts=None,
):
    """Run several circuit groups in one hardware job and save separate files."""
    print("\n=== Real hardware run ===")
    print(f"Backend: {backend.name}")

    job, results, group_slices = submit_grouped_hardware_job(
        job_groups=job_groups,
        backend=backend,
    )

    saved_dirs = {}
    for group in group_slices:
        saved_dirs[group["group_key"]] = save_hardware_results(
            job=job,
            results=results[group["start"]:group["end"]],
            meta_info=group["meta_info"],
            backend=backend,
            shots=group["shots"],
            folder_ts=folder_ts,
            result_filename=group.get("result_filename", DEFAULT_RESULT_FILENAME),
            timing_filename=group.get("timing_filename", DEFAULT_TIMING_FILENAME),
            job_info_filename=group.get("job_info_filename", DEFAULT_JOB_INFO_FILENAME),
            raw_result_filename=group.get("raw_result_filename", DEFAULT_RAW_RESULT_FILENAME),
        )
    return saved_dirs


def submit_hardware_job(transpiled_by_agent, backend, shots):
    """Submit one Sampler job containing one circuit per agent."""
    sampler = Sampler(mode=backend)

    # Request scheduler timing metadata, used later for time-ordering analysis.
    sampler.options.experimental = {"execution": {"scheduler_timing": True}}
    print("Enabled scheduler_timing for Sampler job.")

    circuits = []
    meta_info = []
    for agent_name, tqc in transpiled_by_agent.items():
        circuits.append(tqc)
        meta_info.append(agent_name)

    job = sampler.run(circuits, shots=shots)
    results = job.result()
    return job, results, meta_info


def submit_grouped_hardware_job(job_groups, backend):
    """Submit one Sampler job containing multiple named circuit groups."""
    sampler = Sampler(mode=backend)

    # Each SamplerPub can carry its own shot count, which lets main circuits and
    # accuracy-test circuits share one IBM job.
    sampler.options.experimental = {"execution": {"scheduler_timing": True}}
    print("Enabled scheduler_timing for Sampler job.")

    pubs = []
    group_slices = []

    for group in job_groups:
        start = len(pubs)
        meta_info = []
        for agent_name, tqc in group["transpiled_by_agent"].items():
            pubs.append(SamplerPub(tqc, shots=group["shots"]))
            meta_info.append(agent_name)

        group_slices.append(
            {
                **group,
                "meta_info": meta_info,
                "start": start,
                "end": len(pubs),
            }
        )

    job = sampler.run(pubs)
    results = list(job.result())
    return job, results, group_slices


def save_hardware_results(
    job,
    results,
    meta_info,
    backend,
    shots,
    folder_ts=None,
    result_filename=DEFAULT_RESULT_FILENAME,
    timing_filename=DEFAULT_TIMING_FILENAME,
    job_info_filename=DEFAULT_JOB_INFO_FILENAME,
    raw_result_filename=DEFAULT_RAW_RESULT_FILENAME,
):
    """Save result counts, scheduler metadata, job info, and the raw result."""
    folder_ts, run_folder_name = make_run_folder_name(backend, folder_ts)
    results_dir = DATA_DIR_REAL / run_folder_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Store the IBM job id separately so it is easy to look up later.
    save_json(
        results_dir / job_info_filename,
        {
            "backend": backend.name,
            "job_id": get_job_id(job),
            "shots": int(shots),
            "timestamp": folder_ts,
        },
    )

    run_data = {
        "agents": {},
        "backend": backend.name,
        "kind": "real_hardware_run",
        "shots": int(shots),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    timing_data = {}

    # Keep counts and timing metadata in separate JSON files because they are
    # used by different parts of the analysis.
    for agent_name, pub_res in zip(meta_info, results):
        run_data["agents"][agent_name] = {
            "counts": get_counts_from_sampler_result(pub_res)
        }
        timing_data[agent_name] = extract_pub_result_metadata(pub_res)

    save_json(results_dir / result_filename, run_data)
    save_json(results_dir / timing_filename, timing_data)

    # The raw Sampler result is useful if later analysis needs fields that were
    # not exported into the smaller JSON files.
    with open(results_dir / raw_result_filename, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved real-hardware data to: {results_dir.resolve()}")
    return results_dir


def get_counts_from_sampler_result(pub_res):
    """Extract the single classical register counts from one Sampler result."""
    data = pub_res.data

    if hasattr(data, "keys"):
        reg_names = list(data.keys())
        if len(reg_names) != 1:
            raise ValueError(
                f"Expected exactly 1 classical register, found {len(reg_names)}: {reg_names}"
            )
        return counts_to_jsonable(data[reg_names[0]].get_counts())

    reg_names = [name for name in data.__dict__ if not name.startswith("_")]
    regs_with_counts = []
    for reg_name in reg_names:
        datum = getattr(data, reg_name)
        if hasattr(datum, "get_counts"):
            regs_with_counts.append(reg_name)

    if len(regs_with_counts) != 1:
        raise ValueError(
            "Expected exactly 1 classical register with counts, found "
            f"{len(regs_with_counts)}: {regs_with_counts}"
        )

    return counts_to_jsonable(getattr(data, regs_with_counts[0]).get_counts())


def extract_pub_result_metadata(pub_res):
    """Extract PUB-level metadata returned by SamplerV2."""
    if hasattr(pub_res, "metadata"):
        return to_jsonable(pub_res.metadata)
    return {}


def make_run_folder_name(backend, folder_ts=None):
    """Create the timestamp and shared run-folder name for data and plots."""
    if folder_ts is None:
        folder_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder_ts, f"{backend.name}_{folder_ts}"


def get_job_id(job):
    """Return the IBM job id if the runtime object exposes one."""
    try:
        return job.job_id()
    except Exception:
        return None


def save_json(path: Path, obj):
    """Write one formatted JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def counts_to_jsonable(counts):
    """Convert Qiskit counts to plain JSON keys and integer values."""
    return {str(k): int(v) for k, v in counts.items()}


def to_jsonable(obj):
    """Recursively convert metadata objects into JSON-serializable values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {
            str(k): to_jsonable(v)
            for k, v in obj.__dict__.items()
            if not str(k).startswith("_")
        }
    return str(obj)


if __name__ == "__main__":
    # can be run here to test but usually gets called from the main run.py script
    backend = QiskitRuntimeService().backend("ibm_marrakesh")
    transpiled, folder_ts = prepare_real_hardware_run(backend, save_plots=True)
    run_real_hardware_for_backend(
        backend,
        transpiled,
        shots=1000,
        folder_ts=folder_ts,
    )
