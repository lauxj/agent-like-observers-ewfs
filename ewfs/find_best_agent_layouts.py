"""
find_best_agent_layouts.py
Find the best connected physical-qubit layout for each agent circuit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from ewfs.agents import AGENTS
except ModuleNotFoundError:
    from agents import AGENTS


CALIBRATION_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "IBM_coupling_map"
)

DEFAULT_READOUT_WEIGHT = 1.0
DEFAULT_CZ_WEIGHT = 1.0
DEFAULT_COHERENCE_WEIGHT = 0.3
DEFAULT_M_PRIORITY_FACTOR = 1.5
DEFAULT_CHOICE_DISTANCE_WEIGHT = 0.0015
DEFAULT_FREE_QUBIT_CANDIDATE_LIMIT = 24


@dataclass(frozen=True)
class CalibrationGraph:
    backend_name: str
    readout_error: dict[int, float]
    t1_us: dict[int, float]
    t2_us: dict[int, float]
    adjacency: dict[int, frozenset[int]]
    cz_error: dict[tuple[int, int], float]


@dataclass(frozen=True)
class AgentProblem:
    agent_name: str
    qubit_names: list[str]
    two_qubit_counts: dict[tuple[int, int], int]
    qubit_activity: dict[int, int]
    logical_priority: dict[int, float]
    active_nodes: list[int]
    free_nodes: list[int]
    logical_adjacency: dict[int, frozenset[int]]
    search_order: list[int]


@dataclass(frozen=True)
class LayoutResult:
    layout: list[int]
    total_score: float
    readout_score: float
    cz_score: float
    coherence_score: float
    choice_distance: int | None
    active_mapping: dict[int, int]


_CALIBRATION_GRAPH_CACHE: dict[Path, CalibrationGraph] = {}
_OPTIMAL_LAYOUT_CACHE: dict[tuple, list[int]] = {}
_SHORTEST_PATH_DISTANCE_CACHE: dict[Path, dict[int, dict[int, int]]] = {}


def parse_calibration_filename(csv_path: Path) -> tuple[str, datetime]:
    prefix = "ibm_"
    infix = "_calibrations_"

    stem = csv_path.stem
    if not stem.startswith(prefix) or infix not in stem:
        raise ValueError(
            f"Calibration file name does not match the expected format: {csv_path.name}"
        )

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


def find_latest_calibration_csv(backend_name: str, calibration_dir: Path = CALIBRATION_DIR) -> Path:
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
        stem = csv_path.stem
        suffix = "_calibrations"
        if suffix in stem:
            return stem.split(suffix, 1)[0]
        return stem


def parse_neighbor_metric(cell: object) -> dict[int, float]:
    if pd.isna(cell):
        return {}

    text = str(cell).strip()
    if not text:
        return {}

    out: dict[int, float] = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        neighbor, value = item.split(":", 1)
        out[int(neighbor)] = float(value)
    return out


def load_calibration_graph(csv_path: Path) -> CalibrationGraph:
    df = pd.read_csv(csv_path)

    required = {"Qubit", "Readout assignment error", "T1 (us)", "T2 (us)", "CZ error"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    readout_error: dict[int, float] = {}
    t1_us: dict[int, float] = {}
    t2_us: dict[int, float] = {}
    adjacency: dict[int, set[int]] = defaultdict(set)
    edge_samples: dict[tuple[int, int], list[float]] = defaultdict(list)

    for _, row in df.iterrows():
        qubit = int(row["Qubit"])
        readout_error[qubit] = float(row["Readout assignment error"])
        t1_value = float(row["T1 (us)"])
        t2_value = float(row["T2 (us)"])
        if t1_value <= 0 or t2_value <= 0:
            raise ValueError(
                f"Invalid T1/T2 value for qubit {qubit} in {csv_path.name}: "
                f"T1={t1_value}, T2={t2_value}"
            )
        t1_us[qubit] = t1_value
        t2_us[qubit] = t2_value

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
        t1_us=t1_us,
        t2_us=t2_us,
        adjacency=frozen_adjacency,
        cz_error=cz_error,
    )


def ordered_qubit_names(circuit) -> list[str]:
    names: list[str] = []
    for qreg in circuit.qregs:
        if len(qreg) == 1:
            names.append(qreg.name)
            continue
        for index in range(len(qreg)):
            names.append(f"{qreg.name}[{index}]")
    return names


def collect_two_qubit_counts(circuit) -> dict[tuple[int, int], int]:
    counts: Counter[tuple[int, int]] = Counter()
    global_index = {qubit: circuit.find_bit(qubit).index for qubit in circuit.qubits}

    def walk(subcircuit, qubit_map: dict[object, object]) -> None:
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
                    walk(block, block_map)
                continue

            if len(mapped_qubits) != 2:
                continue
            if operation.name in {"barrier", "measure"}:
                continue

            physical_pair = tuple(sorted(global_index[qubit] for qubit in mapped_qubits))
            counts[physical_pair] += 1

    walk(circuit, {qubit: qubit for qubit in circuit.qubits})
    return dict(counts)


def collect_qubit_activity(circuit) -> dict[int, int]:
    counts: Counter[int] = Counter()
    global_index = {qubit: circuit.find_bit(qubit).index for qubit in circuit.qubits}

    def walk(subcircuit, qubit_map: dict[object, object]) -> None:
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
                    walk(block, block_map)
                continue

            if operation.name in {"barrier", "measure"}:
                continue

            for qubit in mapped_qubits:
                counts[global_index[qubit]] += 1

    walk(circuit, {qubit: qubit for qubit in circuit.qubits})
    return {index: counts.get(index, 0) for index in range(circuit.num_qubits)}


def build_logical_adjacency(
    num_qubits: int,
    two_qubit_counts: dict[tuple[int, int], int],
) -> dict[int, frozenset[int]]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(num_qubits)}
    for left, right in two_qubit_counts:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return {
        index: frozenset(neighbors)
        for index, neighbors in adjacency.items()
    }


def build_search_order(
    active_nodes: list[int],
    logical_adjacency: dict[int, frozenset[int]],
    two_qubit_counts: dict[tuple[int, int], int],
) -> list[int]:
    if not active_nodes:
        return []

    def node_priority(node: int) -> tuple[int, int, int]:
        weighted_degree = sum(
            two_qubit_counts[tuple(sorted((node, neighbor)))]
            for neighbor in logical_adjacency[node]
        )
        return (len(logical_adjacency[node]), weighted_degree, -node)

    root = max(active_nodes, key=node_priority)
    order = [root]
    seen = {root}

    while len(order) < len(active_nodes):
        frontier = {
            neighbor
            for mapped_node in order
            for neighbor in logical_adjacency[mapped_node]
            if neighbor not in seen
        }
        if frontier:
            next_node = max(frontier, key=node_priority)
        else:
            remaining = [node for node in active_nodes if node not in seen]
            next_node = max(remaining, key=node_priority)
        order.append(next_node)
        seen.add(next_node)

    return order


def build_logical_priority(
    qubit_names: list[str],
    m_priority_factor: float,
) -> dict[int, float]:
    priority = {}
    for index, name in enumerate(qubit_names):
        priority[index] = m_priority_factor if name in {"M", "M1"} else 1.0
    return priority


def build_agent_problem(
    agent_name: str,
    build_fn: Callable,
    m_priority_factor: float,
) -> AgentProblem:
    circuit = build_fn()
    return build_agent_problem_from_circuit(
        agent_name=agent_name,
        circuit=circuit,
        m_priority_factor=m_priority_factor,
    )


def build_agent_problem_from_circuit(
    agent_name: str,
    circuit,
    m_priority_factor: float,
) -> AgentProblem:
    qubit_names = ordered_qubit_names(circuit)
    two_qubit_counts = collect_two_qubit_counts(circuit)
    qubit_activity = collect_qubit_activity(circuit)
    logical_priority = build_logical_priority(
        qubit_names=qubit_names,
        m_priority_factor=m_priority_factor,
    )
    logical_adjacency = build_logical_adjacency(circuit.num_qubits, two_qubit_counts)
    active_nodes = sorted({node for edge in two_qubit_counts for node in edge})
    free_nodes = [node for node in range(circuit.num_qubits) if node not in active_nodes]
    search_order = build_search_order(active_nodes, logical_adjacency, two_qubit_counts)

    return AgentProblem(
        agent_name=agent_name,
        qubit_names=qubit_names,
        two_qubit_counts=two_qubit_counts,
        qubit_activity=qubit_activity,
        logical_priority=logical_priority,
        active_nodes=active_nodes,
        free_nodes=free_nodes,
        logical_adjacency=logical_adjacency,
        search_order=search_order,
    )


def find_connected_components(
    logical_adjacency: dict[int, frozenset[int]],
    nodes: list[int],
) -> list[list[int]]:
    remaining = set(nodes)
    components: list[list[int]] = []

    while remaining:
        root = min(remaining)
        stack = [root]
        component = []
        remaining.remove(root)

        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in logical_adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)

        components.append(sorted(component))

    return components


def slice_problem_to_nodes(
    problem: AgentProblem,
    logical_nodes: list[int],
) -> tuple[AgentProblem, dict[int, int]]:
    orig_to_local = {
        logical: index
        for index, logical in enumerate(sorted(logical_nodes))
    }
    local_to_orig = {
        local: logical
        for logical, local in orig_to_local.items()
    }

    two_qubit_counts = {
        tuple(sorted((orig_to_local[left], orig_to_local[right]))): count
        for (left, right), count in problem.two_qubit_counts.items()
        if left in orig_to_local and right in orig_to_local
    }
    logical_adjacency = build_logical_adjacency(len(logical_nodes), two_qubit_counts)
    active_nodes = sorted({node for edge in two_qubit_counts for node in edge})

    subproblem = AgentProblem(
        agent_name=problem.agent_name,
        qubit_names=[problem.qubit_names[local_to_orig[index]] for index in range(len(logical_nodes))],
        two_qubit_counts=two_qubit_counts,
        qubit_activity={
            local: problem.qubit_activity[local_to_orig[local]]
            for local in range(len(logical_nodes))
        },
        logical_priority={
            local: problem.logical_priority[local_to_orig[local]]
            for local in range(len(logical_nodes))
        },
        active_nodes=active_nodes,
        free_nodes=[],
        logical_adjacency=logical_adjacency,
        search_order=build_search_order(active_nodes, logical_adjacency, two_qubit_counts),
    )
    return subproblem, local_to_orig


def get_calibration_graph(csv_path: Path) -> CalibrationGraph:
    resolved_path = csv_path.resolve()
    if resolved_path not in _CALIBRATION_GRAPH_CACHE:
        _CALIBRATION_GRAPH_CACHE[resolved_path] = load_calibration_graph(resolved_path)
    return _CALIBRATION_GRAPH_CACHE[resolved_path]


def get_shortest_path_distances(
    calibration: CalibrationGraph,
    csv_path: Path,
) -> dict[int, dict[int, int]]:
    resolved_path = csv_path.resolve()
    if resolved_path in _SHORTEST_PATH_DISTANCE_CACHE:
        return _SHORTEST_PATH_DISTANCE_CACHE[resolved_path]

    distances: dict[int, dict[int, int]] = {}
    physical_nodes = sorted(calibration.readout_error)

    for source in physical_nodes:
        source_distances = {source: 0}
        frontier = [source]

        while frontier:
            current = frontier.pop(0)
            for neighbor in calibration.adjacency.get(current, frozenset()):
                if neighbor in source_distances:
                    continue
                source_distances[neighbor] = source_distances[current] + 1
                frontier.append(neighbor)

        distances[source] = source_distances

    _SHORTEST_PATH_DISTANCE_CACHE[resolved_path] = distances
    return distances


def build_shortest_path_distances(
    calibration: CalibrationGraph,
) -> dict[int, dict[int, int]]:
    distances: dict[int, dict[int, int]] = {}
    physical_nodes = sorted(calibration.readout_error)

    for source in physical_nodes:
        source_distances = {source: 0}
        frontier = [source]

        while frontier:
            current = frontier.pop(0)
            for neighbor in calibration.adjacency.get(current, frozenset()):
                if neighbor in source_distances:
                    continue
                source_distances[neighbor] = source_distances[current] + 1
                frontier.append(neighbor)

        distances[source] = source_distances

    return distances


def _problem_cache_key(
    backend_name: str,
    csv_path: Path,
    problem: AgentProblem,
    readout_weight: float,
    cz_weight: float,
    coherence_weight: float,
    choice_distance_weight: float,
) -> tuple:
    return (
        backend_name,
        str(csv_path.resolve()),
        tuple(problem.qubit_names),
        tuple(sorted(problem.two_qubit_counts.items())),
        tuple(sorted(problem.qubit_activity.items())),
        tuple(sorted(problem.logical_priority.items())),
        readout_weight,
        cz_weight,
        coherence_weight,
        choice_distance_weight,
    )


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
    csv_path = find_latest_calibration_csv(
        backend_name=backend_name,
        calibration_dir=calibration_dir,
    )
    calibration = get_calibration_graph(csv_path)
    problem = build_agent_problem_from_circuit(
        agent_name=agent_name,
        circuit=circuit,
        m_priority_factor=m_priority_factor,
    )
    cache_key = _problem_cache_key(
        backend_name=backend_name,
        csv_path=csv_path,
        problem=problem,
        readout_weight=readout_weight,
        cz_weight=cz_weight,
        coherence_weight=coherence_weight,
        choice_distance_weight=choice_distance_weight,
    )
    if cache_key not in _OPTIMAL_LAYOUT_CACHE:
        components = find_connected_components(
            logical_adjacency=problem.logical_adjacency,
            nodes=problem.active_nodes,
        )
        if len(components) <= 1:
            result = find_best_layout(
                problem=problem,
                calibration=calibration,
                readout_weight=readout_weight,
                cz_weight=cz_weight,
                coherence_weight=coherence_weight,
                choice_distance_weight=choice_distance_weight,
            )
            _OPTIMAL_LAYOUT_CACHE[cache_key] = result.layout
        else:
            component_order = sorted(
                components,
                key=lambda component: (
                    -sum(
                        problem.logical_priority[node] * max(problem.qubit_activity[node], 1)
                        for node in component
                    ),
                    -len(component),
                    component,
                ),
            )
            used_physical: set[int] = set()
            layout = [-1] * len(problem.qubit_names)

            for component in component_order:
                subproblem, local_to_orig = slice_problem_to_nodes(problem, component)
                result = find_best_layout(
                    problem=subproblem,
                    calibration=calibration,
                    readout_weight=readout_weight,
                    cz_weight=cz_weight,
                    coherence_weight=coherence_weight,
                    choice_distance_weight=choice_distance_weight,
                    initial_used_physical=used_physical,
                )
                for local_index, physical in enumerate(result.layout):
                    original_index = local_to_orig[local_index]
                    layout[original_index] = physical
                    used_physical.add(physical)

            chosen_free = choose_free_qubits(
                used_physical=used_physical,
                problem=problem,
                    calibration=calibration,
                    readout_weight=readout_weight,
                    coherence_weight=coherence_weight,
                    choice_distance_weight=choice_distance_weight,
                    shortest_path_distances=get_shortest_path_distances(calibration, csv_path),
                )
            for logical, physical in chosen_free.items():
                layout[logical] = physical

            _OPTIMAL_LAYOUT_CACHE[cache_key] = layout
    return list(_OPTIMAL_LAYOUT_CACHE[cache_key])


def choose_free_qubits(
    used_physical: set[int],
    problem: AgentProblem,
    calibration: CalibrationGraph,
    readout_weight: float,
    coherence_weight: float,
    choice_distance_weight: float,
    shortest_path_distances: dict[int, dict[int, int]],
    candidate_limit: int = DEFAULT_FREE_QUBIT_CANDIDATE_LIMIT,
) -> dict[int, int]:
    scored_free_nodes = sorted(
        problem.free_nodes,
        key=lambda logical: (-problem.qubit_activity[logical], logical),
    )
    available = sorted(
        qubit
        for qubit in calibration.readout_error
        if qubit not in used_physical
    )
    available = sorted(
        available,
        key=lambda physical: (
            sum(
                readout_weight
                * readout_penalty_for_logical_qubit(
                    problem=problem,
                    logical=logical,
                    calibration=calibration,
                    physical=physical,
                )
                + coherence_weight
                * coherence_penalty_for_logical_qubit(
                    problem=problem,
                    logical=logical,
                    calibration=calibration,
                    physical=physical,
                )
                for logical in scored_free_nodes
            ),
            physical,
        ),
    )[:candidate_limit]
    if len(scored_free_nodes) > 4:
        chosen: dict[int, int] = {}
        remaining = set(available)
        for logical in scored_free_nodes:
            best_physical = min(
                remaining,
                key=lambda physical: (
                    readout_weight
                    * readout_penalty_for_logical_qubit(
                        problem=problem,
                        logical=logical,
                        calibration=calibration,
                        physical=physical,
                    )
                    + coherence_weight
                    * coherence_penalty_for_logical_qubit(
                        problem=problem,
                        logical=logical,
                        calibration=calibration,
                        physical=physical,
                    ),
                    physical,
                ),
            )
            chosen[logical] = best_physical
            remaining.remove(best_physical)
        return chosen

    best_assignment: dict[int, int] | None = None
    best_key = None

    for physical_selection in permutations(available, len(scored_free_nodes)):
        assignment = {
            logical: physical
            for logical, physical in zip(scored_free_nodes, physical_selection)
        }
        readout_cost = sum(
            readout_penalty_for_logical_qubit(
                problem=problem,
                logical=logical,
                calibration=calibration,
                physical=physical,
            )
            for logical, physical in assignment.items()
        )
        coherence_cost = sum(
            coherence_penalty_for_logical_qubit(
                problem=problem,
                logical=logical,
                calibration=calibration,
                physical=physical,
            )
            for logical, physical in assignment.items()
        )
        choice_distance = get_choice_distance(
            problem=problem,
            assignment=assignment,
            shortest_path_distances=shortest_path_distances,
        )
        total_cost = (
            readout_weight * readout_cost
            + coherence_weight * coherence_cost
            - choice_distance_weight * (choice_distance or 0)
        )
        candidate_key = (
            total_cost,
            readout_cost,
            coherence_cost,
            -(choice_distance or -1),
            tuple(assignment[logical] for logical in sorted(assignment)),
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError(f"Could not assign free qubits for {problem.agent_name}.")
    return best_assignment


def coherence_penalty_for_physical_qubit(
    calibration: CalibrationGraph,
    physical: int,
) -> float:
    return 0.5 * (
        (1.0 / calibration.t1_us[physical]) + (1.0 / calibration.t2_us[physical])
    )


def readout_penalty_for_logical_qubit(
    problem: AgentProblem,
    logical: int,
    calibration: CalibrationGraph,
    physical: int,
) -> float:
    return problem.logical_priority[logical] * calibration.readout_error[physical]


def coherence_penalty_for_logical_qubit(
    problem: AgentProblem,
    logical: int,
    calibration: CalibrationGraph,
    physical: int,
) -> float:
    return (
        problem.logical_priority[logical]
        * problem.qubit_activity[logical]
        * coherence_penalty_for_physical_qubit(calibration, physical)
    )


def get_choice_nodes(problem: AgentProblem) -> tuple[int, int] | None:
    name_to_index = {
        name: index
        for index, name in enumerate(problem.qubit_names)
    }
    if "Achoice" in name_to_index and "Bchoice" in name_to_index:
        return name_to_index["Achoice"], name_to_index["Bchoice"]
    return None


def get_choice_distance(
    problem: AgentProblem,
    assignment: dict[int, int],
    shortest_path_distances: dict[int, dict[int, int]],
) -> int | None:
    choice_nodes = get_choice_nodes(problem)
    if choice_nodes is None:
        return None

    left, right = choice_nodes
    if left not in assignment or right not in assignment:
        return None

    left_physical = assignment[left]
    right_physical = assignment[right]
    return shortest_path_distances[left_physical].get(right_physical)


def find_best_layout(
    problem: AgentProblem,
    calibration: CalibrationGraph,
    readout_weight: float,
    cz_weight: float,
    coherence_weight: float,
    choice_distance_weight: float,
    initial_used_physical: set[int] | None = None,
) -> LayoutResult:
    degrees = {
        logical_node: len(problem.logical_adjacency[logical_node])
        for logical_node in problem.active_nodes
    }

    best: LayoutResult | None = None
    physical_nodes = sorted(calibration.readout_error)
    shortest_path_distances = build_shortest_path_distances(calibration)

    def finalize(
        active_mapping: dict[int, int],
        used_physical: set[int],
        active_readout: float,
        cz_cost: float,
        coherence_cost: float,
    ) -> None:
        nonlocal best

        if problem.free_nodes:
            chosen_free = choose_free_qubits(
                used_physical=used_physical,
                problem=problem,
                calibration=calibration,
                readout_weight=readout_weight,
                coherence_weight=coherence_weight,
                choice_distance_weight=choice_distance_weight,
                shortest_path_distances=shortest_path_distances,
            )
        else:
            chosen_free = {}
        free_readout = sum(
            readout_penalty_for_logical_qubit(
                problem=problem,
                logical=logical,
                calibration=calibration,
                physical=physical,
            )
            for logical, physical in chosen_free.items()
        )
        free_coherence = sum(
            coherence_penalty_for_logical_qubit(
                problem=problem,
                logical=logical,
                calibration=calibration,
                physical=physical,
            )
            for logical, physical in chosen_free.items()
        )
        total_readout = active_readout + free_readout
        total_coherence = coherence_cost + free_coherence

        layout = [-1] * len(problem.qubit_names)
        for logical, physical in active_mapping.items():
            layout[logical] = physical
        for logical, physical in chosen_free.items():
            layout[logical] = physical
        choice_distance = get_choice_distance(
            problem=problem,
            assignment={
                logical: physical
                for logical, physical in enumerate(layout)
                if physical != -1
            },
            shortest_path_distances=shortest_path_distances,
        )

        total_score = (
            readout_weight * total_readout
            + cz_weight * cz_cost
            + coherence_weight * total_coherence
            - choice_distance_weight * (choice_distance or 0)
        )
        candidate = LayoutResult(
            layout=layout,
            total_score=total_score,
            readout_score=total_readout,
            cz_score=cz_cost,
            coherence_score=total_coherence,
            choice_distance=choice_distance,
            active_mapping=dict(active_mapping),
        )

        if best is None:
            best = candidate
            return

        current_key = (
            candidate.total_score,
            candidate.readout_score,
            candidate.cz_score,
            candidate.coherence_score,
            -(candidate.choice_distance or -1),
            candidate.layout,
        )
        best_key = (
            best.total_score,
            best.readout_score,
            best.cz_score,
            best.coherence_score,
            -(best.choice_distance or -1),
            best.layout,
        )
        if current_key < best_key:
            best = candidate

    def search(
        index: int,
        active_mapping: dict[int, int],
        used_physical: set[int],
        active_readout: float,
        cz_cost: float,
        coherence_cost: float,
    ) -> None:
        if index == len(problem.search_order):
            finalize(active_mapping, used_physical, active_readout, cz_cost, coherence_cost)
            return

        logical = problem.search_order[index]
        mapped_neighbors = [
            neighbor
            for neighbor in problem.logical_adjacency[logical]
            if neighbor in active_mapping
        ]

        if not mapped_neighbors:
            candidates = physical_nodes
        else:
            candidate_sets = [
                calibration.adjacency.get(active_mapping[neighbor], frozenset())
                for neighbor in mapped_neighbors
            ]
            candidates = sorted(set.intersection(*(set(neighbors) for neighbors in candidate_sets)))

        ranked_candidates = []
        for physical in candidates:
            if physical in used_physical:
                continue
            physical_neighbors = calibration.adjacency.get(physical, frozenset())
            if len(physical_neighbors) < degrees[logical]:
                continue

            unmapped_neighbor_count = sum(
                1
                for neighbor in problem.logical_adjacency[logical]
                if neighbor not in active_mapping
            )
            available_neighbors = physical_neighbors - used_physical
            if len(available_neighbors) < unmapped_neighbor_count:
                continue

            incremental_cz = 0.0
            valid = True
            for neighbor in mapped_neighbors:
                edge = tuple(sorted((physical, active_mapping[neighbor])))
                if edge not in calibration.cz_error:
                    valid = False
                    break
                count = problem.two_qubit_counts[tuple(sorted((logical, neighbor)))]
                incremental_cz += calibration.cz_error[edge] * count

            if not valid:
                continue

            incremental_readout = readout_penalty_for_logical_qubit(
                problem=problem,
                logical=logical,
                calibration=calibration,
                physical=physical,
            )
            incremental_coherence = coherence_penalty_for_logical_qubit(
                problem=problem,
                logical=logical,
                calibration=calibration,
                physical=physical,
            )
            ranked_candidates.append(
                (
                    readout_weight * incremental_readout
                    + cz_weight * incremental_cz
                    + coherence_weight * incremental_coherence,
                    incremental_readout,
                    incremental_cz,
                    incremental_coherence,
                    physical,
                )
            )

        ranked_candidates.sort()

        for _, incremental_readout, incremental_cz, incremental_coherence, physical in ranked_candidates:
            active_mapping[logical] = physical
            used_physical.add(physical)
            search(
                index + 1,
                active_mapping,
                used_physical,
                active_readout + incremental_readout,
                cz_cost + incremental_cz,
                coherence_cost + incremental_coherence,
            )
            used_physical.remove(physical)
            del active_mapping[logical]

    search(
        index=0,
        active_mapping={},
        used_physical=set(initial_used_physical or set()),
        active_readout=0.0,
        cz_cost=0.0,
        coherence_cost=0.0,
    )

    if best is None:
        raise RuntimeError(f"No valid connected layout found for {problem.agent_name}.")
    return best


def format_edge_summary(problem: AgentProblem, result: LayoutResult, calibration: CalibrationGraph) -> list[str]:
    lines = []
    for left, right in sorted(problem.two_qubit_counts):
        physical_edge = tuple(sorted((result.layout[left], result.layout[right])))
        lines.append(
            f"{problem.qubit_names[left]}-{problem.qubit_names[right]} "
            f"-> {physical_edge[0]}-{physical_edge[1]} "
            f"(count={problem.two_qubit_counts[(left, right)]}, "
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
        for index, name in enumerate(problem.qubit_names):
            print(f"    {name} -> {result.layout[index]}")
        print("  used couplings:")
        for line in format_edge_summary(problem, result, calibration):
            print(f"    {line}")
        print()

    print("Copy-paste friendly layouts by agent:")
    print("{")
    for problem in problems:
        print(f'  "{problem.agent_name}": {results[problem.agent_name].layout},')
    print("}")
    print()

    size_buckets: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
    for problem in problems:
        size_buckets[len(problem.qubit_names)].append(
            (problem.agent_name, results[problem.agent_name].layout)
        )

    print("Compatibility snippet for the current size-based ibm_transpilation.py table:")
    print("{")
    for qubit_count in sorted(size_buckets):
        agent_name, layout = size_buckets[qubit_count][0]
        print(f"  {qubit_count}: {layout},  # {agent_name}")
    print("}")

    conflicts = [
        (qubit_count, entries)
        for qubit_count, entries in size_buckets.items()
        if len({tuple(layout) for _, layout in entries}) > 1
    ]
    if conflicts:
        print()
        print("Note:")
        for qubit_count, entries in sorted(conflicts):
            names = ", ".join(name for name, _ in entries)
            print(
                f"  {qubit_count}-qubit agents do not share one identical best layout: {names}"
            )


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

    selected_agents = []
    for agent_name, build_fn in AGENTS:
        if selected_agent_names is not None and agent_name not in selected_agent_names:
            continue
        selected_agents.append(
            build_agent_problem(
                agent_name,
                build_fn,
                m_priority_factor=m_priority_factor,
            )
        )

    if not selected_agents:
        raise ValueError("No agents selected.")

    results = {
        problem.agent_name: find_best_layout(
            problem=problem,
            calibration=calibration,
            readout_weight=readout_weight,
            cz_weight=cz_weight,
            coherence_weight=coherence_weight,
            choice_distance_weight=choice_distance_weight,
        )
        for problem in selected_agents
    }

    emit_results(calibration=calibration, problems=selected_agents, results=results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find the best connected physical-qubit layout for the agent circuits "
            "using IBM calibration CSV data."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Path to a specific IBM calibration CSV file.",
    )
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
        default=1.0,
        help="Weight applied to the summed readout error term.",
    )
    parser.add_argument(
        "--cz-weight",
        type=float,
        default=1.0,
        help="Weight applied to the weighted CZ-error term.",
    )
    parser.add_argument(
        "--coherence-weight",
        type=float,
        default=0.3,
        help=(
            "Weight applied to the T1/T2 coherence penalty term. "
            "The penalty for one placed qubit is activity * 0.5 * (1/T1 + 1/T2)."
        ),
    )
    parser.add_argument(
        "--m-priority-factor",
        type=float,
        default=1.5,
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
        raise FileNotFoundError(
            f"No calibration CSV files found in {CALIBRATION_DIR}."
        )

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
