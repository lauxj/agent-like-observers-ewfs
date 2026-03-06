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
    # A in {1,2} selects c0: A=1 -> c0=0, A=2 -> c0=1
    # B in {1,2} selects c1: B=1 -> c1=0, B=2 -> c1=1
    choice_c0 = "0" if A == 1 else "1"
    choice_c1 = "0" if B == 1 else "1"

    num = 0
    den = 0
    for s, n in counts.items():
        if len(s) != 4:
            raise ValueError(f"Expected 4-bit key, got {s!r}")
        c3, c2, c1, c0 = s[0], s[1], s[2], s[3]  # (SD, Arec, B_choice, A_choice)
        if c1 == choice_c1 and c0 == choice_c0:
            den += n
            num += n * pm(c2) * pm(c3)
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
    path2 = root / "data/data_noiseless_simulation/noiseless_run_2026-03-06T11-09-20_shots10000.json"
    print("\nNoiseless simulation:")
    for agent in ["Betting Agent", "Guessing Agent", "Reflex Agent"]:
        print(agent, "S =", LF_violation(str(path2), agent=agent))

    print()
    #fake_hardware test
    print("fake hardware simulation:")
    path = root / "data/data_fake_hardware/ibm_torino_20260305_155645/fake_hardware_noise_sim.json"
    for agent in ["Betting Agent", "Guessing Agent", "Reflex Agent"]:
        print(agent,"S =", LF_violation(str(path), agent=agent))

    # real hardware
    path3 = root / "data/data_real_hardware/ibm_torino_20260305_144515/real_hardware_run.json"
    print("\nReal hardware:")
    for agent in ["Betting Agent", "Guessing Agent", "Reflex Agent"]:
        print(agent, "S =", LF_violation(str(path3), agent=agent))

     #----------------
    print()
    #fake_hardware test
    print("fake hardware simulation:")
    path = root / "data/data_fake_hardware/ibm_torino_20260305_155811/fake_hardware_noise_sim.json"
    for agent in ["Betting Agent", "Guessing Agent", "Reflex Agent"]:
        print(agent,"S =", LF_violation(str(path), agent=agent))

    # real hardware
    path3 = root / "data/data_real_hardware/ibm_torino_20260305_144642/real_hardware_run.json"
    print("\nReal hardware:")
    for agent in ["Betting Agent", "Guessing Agent", "Reflex Agent"]:
        print(agent, "S =", LF_violation(str(path3), agent=agent))