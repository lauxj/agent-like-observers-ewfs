import json
from pathlib import Path


def pm(bit):
    # 0 -> +1, 1 -> -1
    return 1 if bit == "0" else -1


def load(path, agent="Betting"):
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

        c4, c3, c2, c1, c0 = s[0], s[1], s[2], s[3], s[4]

        if c1 == choice_c1 and c0 == choice_c0:
            den += n

            A_bit = c2 if A == 1 else c3
            B_bit = c4

            num += n * pm(A_bit) * pm(B_bit)

    if den == 0:
        raise ValueError(f"No events with c1={choice_c1} and c0={choice_c0}")

    return num / den


def S(counts):
    E11 = E(counts, 1, 1)
    E12 = E(counts, 1, 2)
    E21 = E(counts, 2, 1)
    E22 = E(counts, 2, 2)
    return -E11 + E12 - E21 - E22 - 2


def LF_violation(path, agent="Betting"):
    return S(load(path, agent=agent))


if __name__ == "__main__":

    root = Path(__file__).resolve().parents[1]  # project root (one level above ewfs/)
    # Noiseless simulation test
    path2 = root / "data/data_noiseless_simulation/noiseless_run_2026-03-06T15-47-39_shots10000.json"
    print("\nNoiseless simulation:")
    for agent in ["Betting Agent", "Guessing Agent", "Reflex Agent"]:
        print(agent, "S =", LF_violation(str(path2), agent=agent))

    print()
    #fake_hardware test
    print("fake hardware simulation:")
    path = root / "data/data_fake_hardware/ibm_torino_20260306_155454/fake_hardware_noise_sim.json"
    for agent in ["Betting Agent", "Guessing Agent", "Reflex Agent"]:
        print(agent,"S =", LF_violation(str(path), agent=agent))

    # real hardware
    path3 = root / "data/data_real_hardware/ibm_torino_20260306_155642/processed_results.json"
    print("\nReal hardware:")
    for agent in ["Betting Agent", "Guessing Agent", "Reflex Agent"]:
        print(agent, "S =", LF_violation(str(path3), agent=agent))

