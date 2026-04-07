"""
Find readable, connected qubit layouts for the agent circuits.

The script does four things:
1. Load one IBM calibration CSV into a hardware graph.
2. Summarize one circuit as a small logical interaction graph.
3. Search for the best connected placement of the logical qubits that take part
   in two-qubit gates.
4. Place the remaining "free" qubits on good leftover hardware qubits.

The implementation is intentionally direct and biased toward readability over
micro-optimizations. The current agent circuits are tiny, so a straightforward
search is practical and much easier to audit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

try:
    from ewfs.agents import AGENTS
except ModuleNotFoundError:
    from agents import AGENTS


CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "data" / "IBM_coupling_map"

DEFAULT_READOUT_WEIGHT = 1.0
DEFAULT_CZ_WEIGHT = 1.0
DEFAULT_COHERENCE_WEIGHT = 0.3
DEFAULT_M_PRIORITY_FACTOR = 1.5
DEFAULT_CHOICE_DISTANCE_WEIGHT = 0.0015
FREE_QUBIT_POOL_SIZE = 12


@dataclass(frozen=True)
class CalibrationGraph:
    backend_name: str
    readout_error: dict[int, float]
    coherence_penalty: dict[int, float]
    adjacency: dict[int, frozenset[int]]
    cz_error: dict[tuple[int, int], float]


@dataclass(frozen=True)
class AgentProblem:
    agent_name: str
    qubit_names: list[str]
    edge_counts: dict[tuple[int, int], int]
    qubit_activity: dict[int, int]
    logical_priority: dict[int, float]
    logical_adjacency: dict[int, frozenset[int]]
    active_nodes: list[int]
    free_nodes: list[int]
    search_order: list[int]


@dataclass(frozen=True)
class LayoutResult:
    layout: list[int]
    total_score: float
    readout_score: float
    cz_score: float
    coherence_score: float
    choice_distance: int | None


@dataclass(frozen=True)
class ScoringWeights:
    readout: float
    cz: float
    coherence: float
    choice_distance: float


def parse_calibration_filename(csv_path: Path) -> tuple[str, datetime]:
    stem = csv_path.stem
    prefix = "ibm_"
    infix = "_calibrations_"

    if not stem.startswith(prefix) or infix not in stem:
        raise ValueError(f"Unexpected calibration file name: {csv_path.name}")

    backend_name, timestamp_text = stem.split(infix, 1)
    timestamp = datetime.strptime(timestamp_text, "%Y-%m-%dT%H_%M_%SZ")
    return backend_name, timestamp


def find_latest_calibration_csvs(calibration_dir: Path) -> dict[str, Path]:
    latest_by_backend: dict[str, tuple[datetime, Path]] = {}

    for csv_path in sorted(calibration_dir.glob("ibm_*_calibrations_*.csv")):
        backend_name, timestamp = parse_calibration_filename(csv_path)
        current = latest_by_backend.get(backend_name)
        if current is None or timestamp > current[0]:
            latest_by_backend[backend_name] = (timestamp, csv_path)

    return {
        backend_name: csv_path
        for backend_name, (_, csv_path) in latest_by_backend.items()
    }


def find_latest_calibration_csv(
    backend_name: str,
    calibration_dir: Path = CALIBRATION_DIR,
) -> Path:
    latest_csvs = find_latest_calibration_csvs(calibration_dir)
    try:
        return latest_csvs[backend_name]
    except KeyError as exc:
        known = ", ".join(sorted(latest_csvs))
        raise ValueError(
            f"No calibration CSV found for backend '{backend_name}'. Known backends: {known}."
        ) from exc


def infer_backend_name(csv_path: Path) -> str:
    try:
        backend_name, _ = parse_calibration_filename(csv_path)
        return backend_name
    except ValueError:
        return csv_path.stem.split("_calibrations", 1)[0]


def parse_neighbor_metric(cell: object) -> dict[int, float]:
    if pd.isna(cell):
        return {}

    text = str(cell).strip()
    if not text:
        return {}

    values: dict[int, float] = {}
    for item in text.split(";"):
        neighbor_text, value_text = item.split(":", 1)
        values[int(neighbor_text.strip())] = float(value_text.strip())
    return values


def load_calibration_graph(csv_path: Path) -> CalibrationGraph:
    df = pd.read_csv(csv_path)
    required = {"Qubit", "Readout assignment error", "T1 (us)", "T2 (us)", "CZ error"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    readout_error: dict[int, float] = {}
    coherence_penalty: dict[int, float] = {}
    adjacency: dict[int, set[int]] = defaultdict(set)
    edge_samples: dict[tuple[int, int], list[float]] = defaultdict(list)

    for _, row in df.iterrows():
        qubit = int(row["Qubit"])
        t1 = float(row["T1 (us)"])
        t2 = float(row["T2 (us)"])
        if t1 <= 0 or t2 <= 0:
            raise ValueError(
                f"Invalid T1/T2 value for qubit {qubit} in {csv_path.name}: T1={t1}, T2={t2}"
            )

        readout_error[qubit] = float(row["Readout assignment error"])
        coherence_penalty[qubit] = 0.5 * ((1.0 / t1) + (1.0 / t2))
        adjacency.setdefault(qubit, set())

        for neighbor, value in parse_neighbor_metric(row["CZ error"]).items():
            edge = tuple(sorted((qubit, neighbor)))
            edge_samples[edge].append(value)
            adjacency[qubit].add(neighbor)
            adjacency[neighbor].add(qubit)

    cz_error = {
        edge: sum(samples) / len(samples)
        for edge, samples in edge_samples.items()
    }
    frozen_adjacency = {
        qubit: frozenset(neighbors)
        for qubit, neighbors in adjacency.items()
    }

    return CalibrationGraph(
        backend_name=infer_backend_name(csv_path),
        readout_error=readout_error,
        coherence_penalty=coherence_penalty,
        adjacency=frozen_adjacency,
        cz_error=cz_error,
    )


def ordered_qubit_names(circuit) -> list[str]:
    names: list[str] = []
    for qreg in circuit.qregs:
        if len(qreg) == 1:
            names.append(qreg.name)
        else:
            for index in range(len(qreg)):
                names.append(f"{qreg.name}[{index}]")
    return names


def iter_operation_qubit_indices(circuit) -> Iterable[list[int]]:
    global_index = {qubit: circuit.find_bit(qubit).index for qubit in circuit.qubits}

    def walk(subcircuit, qubit_map: dict[object, object]) -> Iterable[list[int]]:
        for instruction in subcircuit.data:
            operation = instruction.operation
            mapped_qubits = [qubit_map[qubit] for qubit in instruction.qubits]
            blocks = getattr(operation, "blocks", ())

            if blocks:
                for block in blocks:
                    if len(block.qubits) != len(mapped_qubits):
                        raise ValueError(
                            f"Unsupported control-flow qubit mapping in {operation.name}."
                        )
                    block_map = {
                        block_qubit: mapped_qubits[index]
                        for index, block_qubit in enumerate(block.qubits)
                    }
                    yield from walk(block, block_map)
                continue

            if operation.name in {"barrier", "measure"}:
                continue

            yield [global_index[qubit] for qubit in mapped_qubits]

    yield from walk(circuit, {qubit: qubit for qubit in circuit.qubits})


def summarize_circuit(circuit) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
    edge_counts: Counter[tuple[int, int]] = Counter()
    activity: Counter[int] = Counter()

    for indices in iter_operation_qubit_indices(circuit):
        for index in indices:
            activity[index] += 1

        if len(indices) == 2:
            edge = tuple(sorted(indices))
            edge_counts[edge] += 1
            continue

        if len(indices) > 2:
            raise ValueError(
                "This layout search expects only one- and two-qubit operations after "
                f"expanding control-flow blocks, but found an operation on {len(indices)} qubits."
            )

    return dict(edge_counts), dict(activity)


def build_logical_adjacency(
    num_qubits: int,
    edge_counts: dict[tuple[int, int], int],
) -> dict[int, frozenset[int]]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(num_qubits)}
    for left, right in edge_counts:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return {
        logical: frozenset(neighbors)
        for logical, neighbors in adjacency.items()
    }


def build_search_order(
    active_nodes: list[int],
    logical_adjacency: dict[int, frozenset[int]],
    edge_counts: dict[tuple[int, int], int],
) -> list[int]:
    if not active_nodes:
        return []

    def priority(node: int) -> tuple[int, int, int]:
        weighted_degree = sum(
            edge_counts[tuple(sorted((node, neighbor)))]
            for neighbor in logical_adjacency[node]
        )
        return (len(logical_adjacency[node]), weighted_degree, -node)

    order: list[int] = []
    seen: set[int] = set()
    next_root = max(active_nodes, key=priority)

    while len(order) < len(active_nodes):
        order.append(next_root)
        seen.add(next_root)

        frontier = {
            neighbor
            for node in order
            for neighbor in logical_adjacency[node]
            if neighbor not in seen
        }
        if frontier:
            next_root = max(frontier, key=priority)
            continue

        remaining = [node for node in active_nodes if node not in seen]
        if not remaining:
            break
        next_root = max(remaining, key=priority)

    return order


def build_logical_priority(
    qubit_names: list[str],
    m_priority_factor: float,
) -> dict[int, float]:
    return {
        index: (m_priority_factor if name in {"M", "M1"} else 1.0)
        for index, name in enumerate(qubit_names)
    }


def build_agent_problem(
    agent_name: str,
    build_fn: Callable,
    m_priority_factor: float,
) -> AgentProblem:
    return build_agent_problem_from_circuit(
        agent_name=agent_name,
        circuit=build_fn(),
        m_priority_factor=m_priority_factor,
    )


def build_agent_problem_from_circuit(
    agent_name: str,
    circuit,
    m_priority_factor: float,
) -> AgentProblem:
    qubit_names = ordered_qubit_names(circuit)
    edge_counts, activity = summarize_circuit(circuit)
    logical_adjacency = build_logical_adjacency(circuit.num_qubits, edge_counts)
    active_nodes = sorted({node for edge in edge_counts for node in edge})
    free_nodes = [node for node in range(circuit.num_qubits) if node not in active_nodes]
    search_order = build_search_order(active_nodes, logical_adjacency, edge_counts)

    qubit_activity = {
        logical: activity.get(logical, 0)
        for logical in range(circuit.num_qubits)
    }

    return AgentProblem(
        agent_name=agent_name,
        qubit_names=qubit_names,
        edge_counts=edge_counts,
        qubit_activity=qubit_activity,
        logical_priority=build_logical_priority(qubit_names, m_priority_factor),
        logical_adjacency=logical_adjacency,
        active_nodes=active_nodes,
        free_nodes=free_nodes,
        search_order=search_order,
    )


def shortest_path_distances(calibration: CalibrationGraph) -> dict[int, dict[int, int]]:
    distances: dict[int, dict[int, int]] = {}

    for source in sorted(calibration.readout_error):
        source_distances = {source: 0}
        queue = deque([source])

        while queue:
            current = queue.popleft()
            for neighbor in calibration.adjacency.get(current, frozenset()):
                if neighbor in source_distances:
                    continue
                source_distances[neighbor] = source_distances[current] + 1
                queue.append(neighbor)

        distances[source] = source_distances

    return distances


def readout_cost(
    problem: AgentProblem,
    logical: int,
    calibration: CalibrationGraph,
    physical: int,
) -> float:
    return problem.logical_priority[logical] * calibration.readout_error[physical]


def coherence_cost(
    problem: AgentProblem,
    logical: int,
    calibration: CalibrationGraph,
    physical: int,
) -> float:
    return (
        problem.logical_priority[logical]
        * problem.qubit_activity[logical]
        * calibration.coherence_penalty[physical]
    )


def total_layout_score(
    readout_score: float,
    cz_score: float,
    coherence_score: float,
    choice_distance: int | None,
    weights: ScoringWeights,
) -> float:
    return (
        weights.readout * readout_score
        + weights.cz * cz_score
        + weights.coherence * coherence_score
        - weights.choice_distance * (choice_distance or 0)
    )


def layout_sort_key(result: LayoutResult) -> tuple:
    return (
        result.total_score,
        result.readout_score,
        result.cz_score,
        result.coherence_score,
        -(result.choice_distance or -1),
        result.layout,
    )


def choice_nodes(problem: AgentProblem) -> tuple[int, int] | None:
    name_to_index = {name: index for index, name in enumerate(problem.qubit_names)}
    if "Achoice" in name_to_index and "Bchoice" in name_to_index:
        return name_to_index["Achoice"], name_to_index["Bchoice"]
    return None


def choice_distance_for_mapping(
    problem: AgentProblem,
    mapping: dict[int, int],
    distances: dict[int, dict[int, int]],
) -> int | None:
    pair = choice_nodes(problem)
    if pair is None:
        return None

    left, right = pair
    if left not in mapping or right not in mapping:
        return None

    return distances[mapping[left]].get(mapping[right])


def best_free_qubit_assignment(
    problem: AgentProblem,
    calibration: CalibrationGraph,
    weights: ScoringWeights,
    active_mapping: dict[int, int],
    distances: dict[int, dict[int, int]],
) -> tuple[dict[int, int], float, float, int | None]:
    if not problem.free_nodes:
        return {}, 0.0, 0.0, choice_distance_for_mapping(problem, active_mapping, distances)

    free_nodes = sorted(
        problem.free_nodes,
        key=lambda logical: (-problem.qubit_activity[logical], logical),
    )
    available = sorted(
        physical
        for physical in calibration.readout_error
        if physical not in active_mapping.values()
    )

    # Free qubits never appear in two-qubit gates, so we only need a small pool
    # of individually good leftover qubits before checking all assignments.
    pool = sorted(
        available,
        key=lambda physical: (
            sum(
                readout_cost(problem, logical, calibration, physical)
                + coherence_cost(problem, logical, calibration, physical)
                for logical in free_nodes
            ),
            physical,
        ),
    )[: min(FREE_QUBIT_POOL_SIZE, len(available))]

    best_assignment: dict[int, int] | None = None
    best_key: tuple | None = None
    best_choice_distance: int | None = None
    best_readout = 0.0
    best_coherence = 0.0

    for selected in permutations(pool, len(free_nodes)):
        assignment = dict(zip(free_nodes, selected))
        full_mapping = dict(active_mapping)
        full_mapping.update(assignment)

        assignment_readout = sum(
            readout_cost(problem, logical, calibration, physical)
            for logical, physical in assignment.items()
        )
        assignment_coherence = sum(
            coherence_cost(problem, logical, calibration, physical)
            for logical, physical in assignment.items()
        )
        assignment_choice_distance = choice_distance_for_mapping(
            problem,
            full_mapping,
            distances,
        )
        assignment_total = (
            weights.readout * assignment_readout
            + weights.coherence * assignment_coherence
            - weights.choice_distance * (assignment_choice_distance or 0)
        )
        candidate_key = (
            assignment_total,
            assignment_readout,
            assignment_coherence,
            -(assignment_choice_distance or -1),
            [assignment[logical] for logical in free_nodes],
        )

        if best_key is None or candidate_key < best_key:
            best_assignment = assignment
            best_key = candidate_key
            best_choice_distance = assignment_choice_distance
            best_readout = assignment_readout
            best_coherence = assignment_coherence

    if best_assignment is None:
        raise RuntimeError(f"Could not assign free qubits for {problem.agent_name}.")

    return best_assignment, best_readout, best_coherence, best_choice_distance


def candidate_physical_nodes(
    logical: int,
    problem: AgentProblem,
    calibration: CalibrationGraph,
    mapping: dict[int, int],
    used_physical: set[int],
    weights: ScoringWeights,
) -> list[tuple[float, float, float, float, int]]:
    mapped_neighbors = [
        neighbor
        for neighbor in problem.logical_adjacency[logical]
        if neighbor in mapping
    ]

    if not mapped_neighbors:
        candidates = sorted(calibration.readout_error)
    else:
        candidate_sets = [
            set(calibration.adjacency.get(mapping[neighbor], frozenset()))
            for neighbor in mapped_neighbors
        ]
        candidates = sorted(set.intersection(*candidate_sets))

    ranked: list[tuple[float, float, float, float, int]] = []
    unmapped_neighbor_count = sum(
        1
        for neighbor in problem.logical_adjacency[logical]
        if neighbor not in mapping
    )

    for physical in candidates:
        if physical in used_physical:
            continue

        available_neighbors = calibration.adjacency.get(physical, frozenset()) - used_physical
        if len(available_neighbors) < unmapped_neighbor_count:
            continue

        incremental_cz = 0.0
        for neighbor in mapped_neighbors:
            edge = tuple(sorted((physical, mapping[neighbor])))
            count = problem.edge_counts[tuple(sorted((logical, neighbor)))]
            incremental_cz += calibration.cz_error[edge] * count

        incremental_readout = readout_cost(problem, logical, calibration, physical)
        incremental_coherence = coherence_cost(problem, logical, calibration, physical)
        weighted_total = (
            weights.readout * incremental_readout
            + weights.cz * incremental_cz
            + weights.coherence * incremental_coherence
        )
        ranked.append(
            (
                weighted_total,
                incremental_readout,
                incremental_cz,
                incremental_coherence,
                physical,
            )
        )

    ranked.sort()
    return ranked


def find_best_layout(
    problem: AgentProblem,
    calibration: CalibrationGraph,
    weights: ScoringWeights,
) -> LayoutResult:
    distances = shortest_path_distances(calibration)
    best_result: LayoutResult | None = None

    def finalize(
        active_mapping: dict[int, int],
        active_readout: float,
        active_cz: float,
        active_coherence: float,
    ) -> None:
        nonlocal best_result

        free_mapping, free_readout, free_coherence, free_choice_distance = best_free_qubit_assignment(
            problem=problem,
            calibration=calibration,
            weights=weights,
            active_mapping=active_mapping,
            distances=distances,
        )

        layout = [-1] * len(problem.qubit_names)
        for logical, physical in active_mapping.items():
            layout[logical] = physical
        for logical, physical in free_mapping.items():
            layout[logical] = physical

        total_readout = active_readout + free_readout
        total_coherence = active_coherence + free_coherence
        total_score = total_layout_score(
            readout_score=total_readout,
            cz_score=active_cz,
            coherence_score=total_coherence,
            choice_distance=free_choice_distance,
            weights=weights,
        )

        result = LayoutResult(
            layout=layout,
            total_score=total_score,
            readout_score=total_readout,
            cz_score=active_cz,
            coherence_score=total_coherence,
            choice_distance=free_choice_distance,
        )

        if best_result is None or layout_sort_key(result) < layout_sort_key(best_result):
            best_result = result

    def search(
        index: int,
        mapping: dict[int, int],
        used_physical: set[int],
        readout_score: float,
        cz_score: float,
        coherence_score: float,
    ) -> None:
        if index == len(problem.search_order):
            finalize(mapping, readout_score, cz_score, coherence_score)
            return

        logical = problem.search_order[index]
        for _, extra_readout, extra_cz, extra_coherence, physical in candidate_physical_nodes(
            logical=logical,
            problem=problem,
            calibration=calibration,
            mapping=mapping,
            used_physical=used_physical,
            weights=weights,
        ):
            mapping[logical] = physical
            used_physical.add(physical)
            search(
                index + 1,
                mapping,
                used_physical,
                readout_score + extra_readout,
                cz_score + extra_cz,
                coherence_score + extra_coherence,
            )
            used_physical.remove(physical)
            del mapping[logical]

    if not problem.active_nodes:
        finalize({}, 0.0, 0.0, 0.0)
    else:
        search(
            index=0,
            mapping={},
            used_physical=set(),
            readout_score=0.0,
            cz_score=0.0,
            coherence_score=0.0,
        )

    if best_result is None:
        raise RuntimeError(f"No valid connected layout found for {problem.agent_name}.")
    return best_result


def format_edge_summary(
    problem: AgentProblem,
    result: LayoutResult,
    calibration: CalibrationGraph,
) -> list[str]:
    lines = []
    for left, right in sorted(problem.edge_counts):
        physical_edge = tuple(sorted((result.layout[left], result.layout[right])))
        lines.append(
            f"{problem.qubit_names[left]}-{problem.qubit_names[right]} "
            f"-> {physical_edge[0]}-{physical_edge[1]} "
            f"(count={problem.edge_counts[(left, right)]}, "
            f"cz_error={calibration.cz_error[physical_edge]:.6f})"
        )
    return lines


def emit_results(
    calibration: CalibrationGraph,
    problems: list[AgentProblem],
    results: dict[str, LayoutResult],
) -> None:
    print(f"Backend: {calibration.backend_name}")
    print(f"Physical qubits: {len(calibration.readout_error)}")
    print(f"Coupling edges: {len(calibration.cz_error)}")
    print()

    for problem in problems:
        result = results[problem.agent_name]
        print(problem.agent_name)
        print(f"  qubit order: {problem.qubit_names}")
        print(f"  best layout: {result.layout}")
        print(
            "  score: "
            f"{result.total_score:.8f} "
            f"(readout={result.readout_score:.8f}, "
            f"weighted_cz={result.cz_score:.8f}, "
            f"coherence={result.coherence_score:.8f}, "
            f"choice_distance={result.choice_distance})"
        )
        print("  mapping:")
        for logical, name in enumerate(problem.qubit_names):
            print(f"    {name} -> {result.layout[logical]}")
        print("  used couplings:")
        for line in format_edge_summary(problem, result, calibration):
            print(f"    {line}")
        print()

    print("Copy-paste friendly layouts by agent:")
    print("{")
    for problem in problems:
        print(f'  "{problem.agent_name}": {results[problem.agent_name].layout},')
    print("}")


def find_optimal_layout_for_circuit(
    agent_name: str,
    circuit,
    backend_name: str,
    calibration_dir: Path = CALIBRATION_DIR,
    readout_weight: float = DEFAULT_READOUT_WEIGHT,
    cz_weight: float = DEFAULT_CZ_WEIGHT,
    coherence_weight: float = DEFAULT_COHERENCE_WEIGHT,
    m_priority_factor: float = DEFAULT_M_PRIORITY_FACTOR,
    choice_distance_weight: float = DEFAULT_CHOICE_DISTANCE_WEIGHT,
) -> list[int]:
    csv_path = find_latest_calibration_csv(backend_name, calibration_dir)
    calibration = load_calibration_graph(csv_path)
    problem = build_agent_problem_from_circuit(agent_name, circuit, m_priority_factor)
    weights = ScoringWeights(
        readout=readout_weight,
        cz=cz_weight,
        coherence=coherence_weight,
        choice_distance=choice_distance_weight,
    )
    return find_best_layout(problem, calibration, weights).layout


def run_backend(
    csv_path: Path,
    selected_agent_names: set[str] | None,
    readout_weight: float,
    cz_weight: float,
    coherence_weight: float,
    m_priority_factor: float,
    choice_distance_weight: float,
) -> None:
    calibration = load_calibration_graph(csv_path)
    weights = ScoringWeights(
        readout=readout_weight,
        cz=cz_weight,
        coherence=coherence_weight,
        choice_distance=choice_distance_weight,
    )

    problems = []
    for agent_name, build_fn in AGENTS:
        if selected_agent_names is not None and agent_name not in selected_agent_names:
            continue
        problems.append(
            build_agent_problem(
                agent_name=agent_name,
                build_fn=build_fn,
                m_priority_factor=m_priority_factor,
            )
        )

    if not problems:
        raise ValueError("No agents selected.")

    results = {
        problem.agent_name: find_best_layout(problem, calibration, weights)
        for problem in problems
    }
    emit_results(calibration, problems, results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find the best connected physical-qubit layout for the agent circuits "
            "using IBM calibration CSV data."
        )
    )
    parser.add_argument("--csv", type=Path, help="Path to a specific IBM calibration CSV file.")
    parser.add_argument(
        "--backend",
        action="append",
        help=(
            "Backend name to analyze, for example ibm_kingston or ibm_torino. "
            "Can be passed multiple times. If omitted, the script uses the latest "
            "CSV for every backend found in data/IBM_coupling_map."
        ),
    )
    parser.add_argument(
        "--agent",
        action="append",
        help="Optional agent name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--readout-weight",
        type=float,
        default=DEFAULT_READOUT_WEIGHT,
        help="Weight applied to the summed readout error term.",
    )
    parser.add_argument(
        "--cz-weight",
        type=float,
        default=DEFAULT_CZ_WEIGHT,
        help="Weight applied to the weighted CZ-error term.",
    )
    parser.add_argument(
        "--coherence-weight",
        type=float,
        default=DEFAULT_COHERENCE_WEIGHT,
        help=(
            "Weight applied to the T1/T2 coherence penalty term. "
            "The penalty for one placed qubit is activity * 0.5 * (1/T1 + 1/T2)."
        ),
    )
    parser.add_argument(
        "--m-priority-factor",
        type=float,
        default=DEFAULT_M_PRIORITY_FACTOR,
        help=(
            "Extra importance multiplier for logical qubits named M or M1. "
            "This boosts their readout and T1/T2 impact during layout search."
        ),
    )
    parser.add_argument(
        "--choice-distance-weight",
        type=float,
        default=DEFAULT_CHOICE_DISTANCE_WEIGHT,
        help=(
            "Reward weight for placing Achoice and Bchoice far apart on the hardware graph. "
            "Larger values push them farther apart while still balancing qubit quality."
        ),
    )
    args = parser.parse_args()

    if args.csv is not None and args.backend is not None:
        raise ValueError("Use either --csv or --backend, not both.")

    selected_agent_names = set(args.agent) if args.agent else None

    if args.csv is not None:
        run_backend(
            csv_path=args.csv,
            selected_agent_names=selected_agent_names,
            readout_weight=args.readout_weight,
            cz_weight=args.cz_weight,
            coherence_weight=args.coherence_weight,
            m_priority_factor=args.m_priority_factor,
            choice_distance_weight=args.choice_distance_weight,
        )
        return

    latest_csvs = find_latest_calibration_csvs(CALIBRATION_DIR)
    if not latest_csvs:
        raise FileNotFoundError(f"No calibration CSV files found in {CALIBRATION_DIR}.")

    requested_backends = args.backend if args.backend else sorted(latest_csvs)
    missing_backends = [backend for backend in requested_backends if backend not in latest_csvs]
    if missing_backends:
        known = ", ".join(sorted(latest_csvs))
        raise ValueError(
            f"No calibration CSV found for backend(s): {', '.join(missing_backends)}. "
            f"Known backends: {known}."
        )

    for index, backend_name in enumerate(requested_backends):
        if index:
            print()
            print("=" * 72)
            print()
        run_backend(
            csv_path=latest_csvs[backend_name],
            selected_agent_names=selected_agent_names,
            readout_weight=args.readout_weight,
            cz_weight=args.cz_weight,
            coherence_weight=args.coherence_weight,
            m_priority_factor=args.m_priority_factor,
            choice_distance_weight=args.choice_distance_weight,
        )


if __name__ == "__main__":
    main()
