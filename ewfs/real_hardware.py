"""
real_hardware.py
Runs a real hardware run on actual IBM hardware.
IBM API token must be stored locally.
"""

from pathlib import Path
import json
from datetime import datetime
import pickle
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.primitives.containers.sampler_pub import SamplerPub
try:
    from .ibm_transpilation import transpile_all_agents, PLOT_DIR as IBM_TRANSPILATION_PLOT_DIR
except ImportError:
    from ibm_transpilation import transpile_all_agents, PLOT_DIR as IBM_TRANSPILATION_PLOT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR_REAL = PROJECT_ROOT / "data" / "data_real_hardware"
DATA_DIR_REAL.mkdir(parents=True, exist_ok=True)
BACKEND_NAME = "ibm_torino"


def make_run_folder_name(backend, folder_ts=None):
    """Create the shared run-folder name used for data and plots."""
    if folder_ts is None:
        folder_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder_ts, f"{backend.name}_{folder_ts}"


def save_json(path: Path, obj):
    """Save JSON file."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


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


def extract_pub_result_metadata(pub_res):
    """Extract PUB-level metadata returned by SamplerV2."""
    if hasattr(pub_res, "metadata"):
        return to_jsonable(pub_res.metadata)
    return {}


def counts_to_jsonable(counts):
    """Convert Qiskit counts dict to JSON-serializable format."""
    return {str(k): int(v) for k, v in counts.items()}


def get_counts_from_sampler_result(pub_res):
    """Extract single-register counts from a SamplerV2 result item."""
    data = pub_res.data

    if hasattr(data, "keys"):
        reg_names = list(data.keys())
        if len(reg_names) != 1:
            raise ValueError(
                f"Expected exactly 1 classical register, found {len(reg_names)}: {reg_names}"
            )
        reg = reg_names[0]
        return counts_to_jsonable(data[reg].get_counts())

    reg_names = [k for k in data.__dict__.keys() if not k.startswith("_")]
    regs_with_counts = []
    for reg in reg_names:
        datum = getattr(data, reg)
        if hasattr(datum, "get_counts"):
            regs_with_counts.append(reg)

    if len(regs_with_counts) != 1:
        raise ValueError(
            f"Expected exactly 1 classical register with counts, found {len(regs_with_counts)}: {regs_with_counts}"
        )

    reg = regs_with_counts[0]
    datum = getattr(data, reg)
    return counts_to_jsonable(datum.get_counts())


def submit_hardware_job(transpiled_by_agent, backend, shots):
    """Submit one job to IBM real hardware containing one circuit per agent."""
    sampler = Sampler(mode=backend)

    sampler.options.experimental = {"execution": {"scheduler_timing": True}}
    print("Enabled scheduler_timing for Sampler job.")

    all_circuits = []
    meta_info = []

    for agent_name, tqc in transpiled_by_agent.items():
        all_circuits.append(tqc)
        meta_info.append(agent_name)

    job = sampler.run(all_circuits, shots=shots)
    results = job.result()
    return job, results, meta_info


def submit_grouped_hardware_job(job_groups, backend):
    """Submit one Sampler job containing multiple circuit groups with per-pub shot counts."""
    sampler = Sampler(mode=backend)

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
        group_slices.append({
            **group,
            "meta_info": meta_info,
            "start": start,
            "end": len(pubs),
        })

    job = sampler.run(pubs)
    results = list(job.result())
    return job, results, group_slices


def save_hardware_results(
    job,
    results,
    meta_info,
    backend,
    shots,
    transpiled_by_agent,
    folder_ts=None,
    result_filename="real_hardware_run.json",
    timing_filename="scheduler_timing_metadata.json",
    job_info_filename="job_info.json",
    raw_result_filename="raw_sampler_result.pkl",
):
    """Save hardware result counts for one backend run."""
    folder_ts, run_folder_name = make_run_folder_name(backend, folder_ts)
    results_dir = DATA_DIR_REAL / run_folder_name
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        job_id = job.job_id()
    except Exception:
        job_id = None

    save_json(
        results_dir / job_info_filename,
        {
            "backend": backend.name,
            "job_id": job_id,
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

    for agent_name, pub_res in zip(meta_info, results):
        counts = get_counts_from_sampler_result(pub_res)
        metadata = extract_pub_result_metadata(pub_res)
        run_data["agents"][agent_name] = {"counts": counts}
        timing_data[agent_name] = metadata

    save_json(results_dir / result_filename, run_data)
    save_json(results_dir / timing_filename, timing_data)

    with open(results_dir / raw_result_filename, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved real-hardware data to: {results_dir.resolve()}")
    return results_dir


def prepare_real_hardware_run(
    backend,
    save_plots=True,
    folder_ts=None,
    agent_builders=None,
    plots_subdir="transpiled_agents",
):
    """Prepare transpiled circuits and matching plot folder for one real-hardware run."""
    folder_ts, run_folder_name = make_run_folder_name(backend, folder_ts)
    plots_dir = IBM_TRANSPILATION_PLOT_DIR / "real_hardware" / run_folder_name / plots_subdir
    transpiled_by_agent = transpile_all_agents(
        backend,
        save_plots=save_plots,
        plots_dir=plots_dir,
        agent_builders=agent_builders,
    )
    return transpiled_by_agent, folder_ts


def run_real_hardware_for_backend(
    backend,
    transpiled_by_agent,
    shots=300,
    folder_ts=None,
    result_filename="real_hardware_run.json",
    timing_filename="scheduler_timing_metadata.json",
    job_info_filename="job_info.json",
    raw_result_filename="raw_sampler_result.pkl",
):
    """Run one real-hardware job for all agents on one backend."""
    print("\n--- Real hardware run ---")
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
        transpiled_by_agent=transpiled_by_agent,
        folder_ts=folder_ts,
        result_filename=result_filename,
        timing_filename=timing_filename,
        job_info_filename=job_info_filename,
        raw_result_filename=raw_result_filename,
    )


def run_grouped_real_hardware_for_backend(
    backend,
    job_groups,
    folder_ts=None,
):
    """Run one real-hardware job for multiple circuit groups and save split result files."""
    print("\n--- Real hardware run ---")
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
            transpiled_by_agent=group["transpiled_by_agent"],
            folder_ts=folder_ts,
            result_filename=group.get("result_filename", "real_hardware_run.json"),
            timing_filename=group.get("timing_filename", "scheduler_timing_metadata.json"),
            job_info_filename=group.get("job_info_filename", "job_info.json"),
            raw_result_filename=group.get("raw_result_filename", "raw_sampler_result.pkl"),
        )
    return saved_dirs


if __name__ == "__main__":
    backend = QiskitRuntimeService().backend(BACKEND_NAME)
    transpiled, folder_ts = prepare_real_hardware_run(backend, save_plots=True)
    run_real_hardware_for_backend(
        backend,
        transpiled,
        shots=300,
        folder_ts=folder_ts,
    )
