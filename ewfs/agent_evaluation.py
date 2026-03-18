import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_REAL = PROJECT_ROOT / "data" / "data_real_hardware"
DATA_DIR_NOISELESS = PROJECT_ROOT / "data" / "data_noiseless_simulation"
DATA_DIR_FAKE = PROJECT_ROOT / "data" / "data_fake_hardware"
PLOTS_ROOT = PROJECT_ROOT / "results" / "plots" / "plots_ibm_transpilation" / "agent_evaluation"

WALLET_STATES = ["00", "01", "10", "11"]
IDEAL_PROBS = {
    "00": 1 / 8,
    "01": 3 / 8,
    "10": 3 / 8,
    "11": 1 / 8,
}
PAYOFFS = {
    "00": -3 / 4,
    "01": -1 / 4,
    "10": 1.0,
    "11": 1.0,
}



def clean_bitstring(bitstring: str) -> str:
    return "".join(ch for ch in str(bitstring) if ch in {"0", "1"})



def find_latest_run(data_dir: Path, result_filename: str) -> Path:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    runs = [
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / result_filename).exists()
    ]
    if not runs:
        raise FileNotFoundError(
            f"No run folders with {result_filename} found in {data_dir.resolve()}"
        )

    return max(runs, key=lambda d: d.stat().st_mtime)



def extract_wallet_counts(counts):
    wallet_counts = {state: 0 for state in WALLET_STATES}
    kept_shots = 0
    excluded_undo_shots = 0

    for bitstring, shots in counts.items():
        b = clean_bitstring(bitstring)

        # Betting agent uses 8 classical bits: c[7]...c[0] in the count string.
        if len(b) < 8:
            raise ValueError(f"Bitstring too short to contain wallet bits: {bitstring}")

        # Alice choice is stored in c[0], i.e. the last bit of the bitstring.
        # c[0] = 1 corresponds to Alice choosing the undo branch (A2), which resets
        # the wallet and therefore should be excluded from the betting statistics.
        alice_choice = b[-1]
        if alice_choice == "1":
            excluded_undo_shots += shots
            continue

        wallet = b[0:2]

        if wallet not in wallet_counts:
            raise ValueError(f"Unexpected wallet state '{wallet}' from bitstring '{bitstring}'")

        wallet_counts[wallet] += shots
        kept_shots += shots

    return wallet_counts, kept_shots, excluded_undo_shots



def counts_to_probabilities(wallet_counts, total):
    return {state: (wallet_counts[state] / total if total > 0 else 0.0) for state in WALLET_STATES}



def expected_payoff(probabilities):
    return sum(probabilities[state] * PAYOFFS[state] for state in WALLET_STATES)



def l1_distance_to_ideal(probabilities):
    return sum(abs(probabilities[state] - IDEAL_PROBS[state]) for state in WALLET_STATES)



def load_backend_result(label: str, data_dir: Path, result_filename: str):
    run_dir = find_latest_run(data_dir, result_filename)
    run_name = run_dir.name

    with open(run_dir / result_filename, "r") as f:
        data = json.load(f)

    counts = data["agents"]["Betting Agent"]["counts"]
    wallet_counts, total_used, excluded_undo_shots = extract_wallet_counts(counts)
    probabilities = counts_to_probabilities(wallet_counts, total_used)

    return {
        "label": label,
        "run_dir": run_dir,
        "run_name": run_name,
        "source_file": result_filename,
        "wallet_counts": wallet_counts,
        "total_shots_used": total_used,
        "excluded_undo_shots": excluded_undo_shots,
        "probabilities": probabilities,
        "expected_payoff": expected_payoff(probabilities),
        "l1_distance": l1_distance_to_ideal(probabilities),
    }



def plot_wallet_probabilities(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    states = WALLET_STATES
    x = np.arange(len(states))
    ideal_values = [IDEAL_PROBS[state] for state in states]

    for ax, result in zip(axes, results):
        measured_values = [result["probabilities"][state] for state in states]
        ax.bar(x, measured_values, width=0.6, label="Observed")
        ax.plot(x, ideal_values, marker="o", linestyle="--", label="Ideal")
        ax.set_xticks(x)
        ax.set_xticklabels(states)
        ax.set_xlabel("Wallet state (W1,W0)")
        ax.set_title(result["label"])
        ax.set_ylim(0.0, 0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.02,
            0.95,
            f"shots = {result['total_shots_used']}\nexcluded = {result['excluded_undo_shots']}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
        )

    axes[0].set_ylabel("Probability")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Betting Agent wallet probabilities conditioned on no undo", y=1.02)
    fig.tight_layout()

    plot_path = output_dir / "betting_agent_wallet_probabilities_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return plot_path



def plot_expected_payoff(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [result["label"] for result in results]
    values = [result["expected_payoff"] for result in results]
    ideal_value = sum(IDEAL_PROBS[state] * PAYOFFS[state] for state in WALLET_STATES)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.6)
    ax.axhline(ideal_value, linestyle="--", label=f"Ideal = {ideal_value:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Expected payoff")
    ax.set_title("Betting Agent expected payoff conditioned on no undo")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = output_dir / "betting_agent_expected_payoff_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return plot_path



def save_summary(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "ideal_probabilities": IDEAL_PROBS,
        "ideal_expected_payoff": sum(IDEAL_PROBS[state] * PAYOFFS[state] for state in WALLET_STATES),
        "payoffs": PAYOFFS,
        "backends": {},
    }

    for result in results:
        summary["backends"][result["label"]] = {
            "run_name": result["run_name"],
            "source_file": result["source_file"],
            "total_shots_used": result["total_shots_used"],
            "excluded_undo_shots": result["excluded_undo_shots"],
            "wallet_counts": result["wallet_counts"],
            "probabilities": result["probabilities"],
            "expected_payoff": result["expected_payoff"],
            "l1_distance_to_ideal": result["l1_distance"],
        }

    summary_path = output_dir / "betting_agent_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path



def print_summary(results):
    ideal_payoff = sum(IDEAL_PROBS[state] * PAYOFFS[state] for state in WALLET_STATES)
    print("Ideal conditional probabilities:")
    for state in WALLET_STATES:
        print(f"  {state}: {IDEAL_PROBS[state]:.4f}")
    print(f"Ideal expected payoff: {ideal_payoff:.4f}\n")

    for result in results:
        print(f"{result['label']} — {result['run_name']}")
        print(f"  Shots used (Alice did not undo): {result['total_shots_used']}")
        print(f"  Excluded undo shots: {result['excluded_undo_shots']}")
        print(f"  Expected payoff: {result['expected_payoff']:.4f}")
        print(f"  L1 distance to ideal: {result['l1_distance']:.4f}")
        print("  Wallet probabilities:")
        for state in WALLET_STATES:
            print(
                f"    {state}: {result['wallet_counts'][state]}  "
                f"(p = {result['probabilities'][state]:.4f})"
            )
        print()



def main():
    results = [
        load_backend_result("Noiseless", DATA_DIR_NOISELESS, "noiseless_simulation.json"),
        load_backend_result("Fake hardware", DATA_DIR_FAKE, "fake_hardware_noise_sim.json"),
        load_backend_result("Real hardware", DATA_DIR_REAL, "real_hardware_run.json"),
    ]

    latest_stamp = max(result["run_name"] for result in results)
    output_dir = PLOTS_ROOT / latest_stamp / "Betting_Agent"
    output_dir.mkdir(parents=True, exist_ok=True)

    prob_plot_path = plot_wallet_probabilities(results, output_dir)
    payoff_plot_path = plot_expected_payoff(results, output_dir)
    summary_path = save_summary(results, output_dir)

    print_summary(results)
    print(f"Saved probability comparison plot to: {prob_plot_path}")
    print(f"Saved expected payoff plot to: {payoff_plot_path}")
    print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()