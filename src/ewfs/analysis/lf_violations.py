"""
lf_violations.py
(this function gets called from the main run.py script)
Calculate LF violations from saved experiment counts

    S = -E11 + E12 - E21 - E22 - 2

The LF inequality is satisfied when S >= 0.

The LF calculation uses the values saved to the classical register::

    c[0] <- AC[0]  Alice setting choice x
    c[1] <- BC[0]  Bob setting choice y
    c[2] <- M1[0]  Alice outcome for x=1
    c[3] <- SA[0]  Alice outcome for x=2
    c[4] <- SB[0]  Bob outcome

"""

import json
from pathlib import Path


AGENT_NAMES = [
    "Betting Agent",
    "Guessing Agent",
    "Reflex Agent",
    "Always 3/4 Agent",
]


def LF_violation_details(path, agent="Betting Agent"):
    """Calculate the LF value S for one agent."""
    # First load the raw measurement counts for one agent
    counts = load_counts(path, agent)

    # Calculate the four correlators needed in the LF expression.
    # E11 means x=1, y=1; E12 means x=1, y=2; and so on
    E11 = E_details(counts, x=1, y=1)
    E12 = E_details(counts, x=1, y=2)
    E21 = E_details(counts, x=2, y=1)
    E22 = E_details(counts, x=2, y=2)

    # LF expression used in the thesis. Positive S means a violation.
    s_value = -E11["value"] + E12["value"] - E21["value"] - E22["value"] - 2

    # Store both the correlator values and how many shots contributed to each
    # correlator, so the saved output can be checked later.
    return {
        "S": s_value,
        "correlators": {
            "E11": E11["value"],
            "E12": E12["value"],
            "E21": E21["value"],
            "E22": E22["value"],
        },
        "correlator_shots": {
            "E11": E11["shots"],
            "E12": E12["shots"],
            "E21": E21["shots"],
            "E22": E22["shots"],
        },
        "num_bitstrings": len(counts),
        "total_shots": int(sum(counts.values())),
        "violation": bool(s_value > 0),
    }


def E_details(counts, x, y):
    """Calculate one correlator E_xy.

    Counts are saved by Qiskit as bitstring keys in the JSON file, for example
    {"00101": 123}. Each key is one measured classical-register outcome, and
    the value is how often that outcome occurred.

    The last five bits of each measured bitstring are:

        c4 c3 c2 c1 c0

    c0 chooses Alice setting: 0 -> x=1, 1 -> x=2
    c1 chooses Bob setting:   0 -> y=1, 1 -> y=2
    c2 is Alice's x=1 outcome
    c3 is Alice's x=2 outcome
    c4 is Bob's outcome for either y=1 or y=2
    """
    # Translate the requested settings x/y into the stored choice bits c0/c1.
    wanted_c0 = "0" if x == 1 else "1"
    wanted_c1 = "0" if y == 1 else "1"

    # numerator sums: shots * Alice_outcome * Bob_outcome
    # denominator counts how many shots used this x/y setting pair.
    numerator = 0
    denominator = 0

    for bitstring, shots in counts.items():
        if len(bitstring) < 5:
            raise ValueError(f"Expected at least a 5-bit key, got {bitstring!r}")

        # Only the last five classical bits are relevant for LF.
        c4 = bitstring[-5]  # Bob's outcome
        c3 = bitstring[-4]  # Alice's outcome if x=2
        c2 = bitstring[-3]  # Alice's outcome if x=1
        c1 = bitstring[-2]  # Bob's setting choice y
        c0 = bitstring[-1]  # Alice's setting choice x

        # Keep only shots where the circuit selected the wanted x/y setting.
        if c0 != wanted_c0 or c1 != wanted_c1:
            continue

        # Select Alice's relevant outcome bit for this x setting.
        alice_bit = c2 if x == 1 else c3
        bob_bit = c4

        # Convert outcome bits to +/-1 and add Alice * Bob.
        numerator += shots * bit_to_pm_one(alice_bit) * bit_to_pm_one(bob_bit)
        denominator += shots

    if denominator == 0:
        raise ValueError(f"No shots found for x={x}, y={y}")

    # E_xy is the average value of Alice_outcome * Bob_outcome.
    return {
        "value": numerator / denominator,
        "shots": int(denominator),
    }


def E(counts, x, y):
    """Return only the correlator value."""
    # Convenience wrapper if only the number is needed.
    return E_details(counts, x, y)["value"]


def bit_to_pm_one(bit):
    """Convert 0 -> +1 and 1 -> -1."""
    return 1 if bit == "0" else -1


def LF_violation(path, agent="Betting Agent"):
    """Save LF results and return one agent's S value."""
    # run.py calls this function when it wants to print one S value.
    # The full LF JSON file is saved at the same time.
    save_lf_violations_for_run(path)
    return LF_violation_details(path, agent)["S"]


def save_lf_violations_for_run(path, agent_names=None):
    """Calculate LF violations for one run file and save them."""
    if agent_names is None:
        agent_names = AGENT_NAMES

    run_path = Path(path)

    # The input path points to one saved experiment file, for example:
    # data/data_fake_hardware/<run_folder>/fake_hardware_noise_sim.json
    # The LF inequality results are saved into the same run folder.
    results = {
        "kind": "lf_violations",
        "source_file": run_path.name,
        "source_path": str(run_path.resolve()),
        "agents": {},
    }

    # Calculate LF data separately for each agent in the result file.
    for agent_name in agent_names:
        results["agents"][agent_name] = LF_violation_details(run_path, agent=agent_name)

    # Save into: <same run folder>/lf_violations/lf_violations.json
    output_dir = run_path.parent / "lf_violations"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "lf_violations.json"
    save_json(output_path, results)

    print(f"Saved LF violations to: {output_path.resolve()}")
    return results


def load_counts(path, agent="Betting Agent"):
    """Load one agent's counts from a saved run JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Saved run files have this structure:
    # {
    #   "agents": {
    #     "Betting Agent": {
    #       "counts": {"00101": 123, "10101": 87, ...}
    #     },
    #     ...
    #   }
    # }
    # So this line selects exactly one agent's counts dictionary.
    return {k: int(v) for k, v in data["agents"][agent]["counts"].items()}


def save_json(path: Path, obj):
    """Write formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# Small helper names used by notebooks / older scripts.
pm = bit_to_pm_one
load = load_counts


def correlator_details(counts):
    return {
        "E11": E_details(counts, x=1, y=1),
        "E12": E_details(counts, x=1, y=2),
        "E21": E_details(counts, x=2, y=1),
        "E22": E_details(counts, x=2, y=2),
    }


def correlators(counts):
    return {key: value["value"] for key, value in correlator_details(counts).items()}


def S(counts):
    corr = correlators(counts)
    return -corr["E11"] + corr["E12"] - corr["E21"] - corr["E22"] - 2
