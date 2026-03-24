"""
lf_violations.py
Calculates LF violations
"""

import json
from pathlib import Path

def pm(bit):
    # 0 -> +1, 1 -> -1
    return 1 if bit == "0" else -1


def load(path, agent="Betting Agent"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: int(v) for k, v in data["agents"][agent]["counts"].items()}


def E(counts, A, B):
    # A in {1,2}: choose A1 (c2) or A2 (c3)
    # B in {1,2}: choose B1 (c1=0) or B2 (c1=1)

    choice_c0 = "0" if A == 1 else "1"  # A_choice
    choice_c1 = "0" if B == 1 else "1"  # B_choice

    num = 0
    den = 0

    for s, n in counts.items():
        if len(s) < 5:
            raise ValueError(f"Expected ≥5-bit key, got {s!r}")

        # Extract the LF-relevant bits (last five classical bits)
        c4, c3, c2, c1, c0 = s[-5], s[-4], s[-3], s[-2], s[-1]

        if c1 == choice_c1 and c0 == choice_c0:
            den += n

            A_bit = c2 if A == 1 else c3
            B_bit = c4

            num += n * pm(A_bit) * pm(B_bit)

    if den == 0:
        raise ValueError(f"No events with c1={choice_c1} and c0={choice_c0}")

    return num / den


def correlators(counts):
    """Return all four correlators entering the LF expression."""
    E11 = E(counts, 1, 1)
    E12 = E(counts, 1, 2)
    E21 = E(counts, 2, 1)
    E22 = E(counts, 2, 2)
    return {
        "E11": E11,
        "E12": E12,
        "E21": E21,
        "E22": E22,
    }



def S(counts):
    corr = correlators(counts)
    return -corr["E11"] + corr["E12"] - corr["E21"] - corr["E22"] - 2


def LF_violation_details(path, agent="Betting Agent"):
    counts = load(path, agent=agent)
    corr = correlators(counts)
    s_value = -corr["E11"] + corr["E12"] - corr["E21"] - corr["E22"] - 2
    total_shots = int(sum(counts.values()))
    return {
        "S": s_value,
        "correlators": corr,
        "num_bitstrings": len(counts),
        "total_shots": total_shots,
        "violation": bool(s_value > 0),
    }



def LF_violation(path, agent="Betting Agent"):
    # Ensure LF results are saved for this run
    save_lf_violations_for_run(path)

    # Return the S value for the requested agent
    return LF_violation_details(path, agent=agent)["S"]


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)



def save_lf_violations_for_run(path, agent_names=None):
    """Compute LF violations for one run file and save them into the same run folder."""
    if agent_names is None:
        agent_names = ["Betting Agent", "Guessing Agent", "Reflex Agent", "Always 3/4 Agent"]

    run_path = Path(path)
    results = {
        "kind": "lf_violations",
        "source_file": str(run_path.name),
        "source_path": str(run_path.resolve()),
        "agents": {},
    }

    for agent in agent_names:
        results["agents"][agent] = LF_violation_details(str(run_path), agent=agent)

    # Create folder inside the run directory to store LF results
    out_dir = run_path.parent / "lf_violations"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "lf_violations.json"
    save_json(out_path, results)

    print(f"Saved LF violations to: {out_path.resolve()}")
    return results


if __name__ == "__main__":

    # For specific runs / Testing:
     root = Path(__file__).resolve().parents[1]  # project root (one level above ewfs/)

     # Noiseless simulation test
     path1 = root / "INSERT PATH HERE FOR TESTING"
     print("\nNoiseless simulation:")
     results1 = save_lf_violations_for_run(path1)
     for agent, values in results1["agents"].items():
         print(agent, "S =", values["S"])

     # fake_hardware test
     print("fake hardware simulation:")
     path2 = root / "INSERT PATH HERE FOR TESTING"
     results2 = save_lf_violations_for_run(path2)
     for agent, values in results2["agents"].items():
         print(agent, "S =", values["S"])

     # real hardware
     print("\nReal hardware:")
     path3 = root / "INSERT PATH HERE FOR TESTING"
     results3 = save_lf_violations_for_run(path3)
     for agent, values in results3["agents"].items():
         print(agent, "S =", values["S"])
