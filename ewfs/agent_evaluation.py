import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_REAL = PROJECT_ROOT / "data" / "data_real_hardware"
DATA_DIR_NOISELESS = PROJECT_ROOT / "data" / "data_noiseless_simulation"
DATA_DIR_FAKE = PROJECT_ROOT / "data" / "data_fake_hardware"
PLOTS_ROOT = PROJECT_ROOT / "results" / "plots" / "plots_agent_evaluation"

IDEAL_COLOR = "#222222"
PAYOFF_BY_WALLET_STATE = {
    "00": -3 / 4,
    "01": -1 / 4,
    "10": 1 / 4,
    "11": 3 / 4,
}
THEORY_C1_PROBABILITIES = {
    "0": 0.5,
    "1": 0.5,
}

STAKE_BY_WALLET_STATE = {
    "00": 3 / 4,
    "01": 1 / 4,
    "10": 3 / 4,
    "11": 1 / 4,
}
# Betting circuit count strings are read as c[7]...c[0].
# M1 is measured into c[2], which is index 5 from the left.
M1_BIT_INDEX_FROM_LEFT = 5

# Guessing circuit count strings are read as c[6]...c[0].
# M2 is measured into c[5] -> index 1 from the left.
# G is measured into c[6] -> index 0 from the left.
GUESSING_M2_INDEX_FROM_LEFT = 1
GUESSING_G_INDEX_FROM_LEFT = 0

# Reflex circuit count strings are read as c[5]...c[0].
# L is measured into c[5] -> index 0 from the left.
# M is measured into c[2] -> index 3 from the left.
REFLEX_L_INDEX_FROM_LEFT = 0
REFLEX_M_INDEX_FROM_LEFT = 3


def clean_bitstring(bitstring: str) -> str:
    return "".join(ch for ch in str(bitstring) if ch in {"0", "1"})


def find_latest_run(data_dir: Path, result_filename: str) -> Path:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    runs = [
        run_dir for run_dir in data_dir.iterdir()
        if run_dir.is_dir() and (run_dir / result_filename).exists()
    ]
    if not runs:
        raise FileNotFoundError(
            f"No run folders with {result_filename} found in {data_dir.resolve()}"
        )

    return max(runs, key=lambda run_dir: run_dir.stat().st_mtime)


def resolve_run_dir(data_dir: Path, result_filename: str, run_name: Optional[str]) -> Tuple[Path, str]:
    if run_name is None:
        return find_latest_run(data_dir, result_filename), "latest"

    run_dir = data_dir / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir.resolve()}")
    if not (run_dir / result_filename).exists():
        raise FileNotFoundError(f"Expected result file not found: {(run_dir / result_filename).resolve()}")

    return run_dir, "manual"


def extract_strategy_probabilities(counts):
    stats = {
        "0": {"shots": 0, "bet_1_4": 0},
        "1": {"shots": 0, "bet_3_4": 0},
    }

    for bitstring, shots in counts.items():
        cleaned = clean_bitstring(bitstring)
        if len(cleaned) < 8:
            raise ValueError(f"Bitstring too short to contain M1 and wallet bits: {bitstring}")

        # Exclude undo branch: Alice chose A2.
        if cleaned[-1] == "1":
            continue

        c1 = cleaned[M1_BIT_INDEX_FROM_LEFT]
        wallet = cleaned[0:2]
        if c1 not in {"0", "1"}:
            raise ValueError(f"Unexpected M1 value '{c1}' from bitstring '{bitstring}'")
        if wallet not in STAKE_BY_WALLET_STATE:
            raise ValueError(f"Unexpected wallet state '{wallet}' from bitstring '{bitstring}'")

        stake = STAKE_BY_WALLET_STATE[wallet]
        stats[c1]["shots"] += shots
        if c1 == "0" and np.isclose(stake, 1 / 4):
            stats[c1]["bet_1_4"] += shots
        if c1 == "1" and np.isclose(stake, 3 / 4):
            stats[c1]["bet_3_4"] += shots

    p_bet_1_4_given_c1_0 = (
        stats["0"]["bet_1_4"] / stats["0"]["shots"] if stats["0"]["shots"] else 0.0
    )
    p_bet_3_4_given_c1_1 = (
        stats["1"]["bet_3_4"] / stats["1"]["shots"] if stats["1"]["shots"] else 0.0
    )

    return {
        "P(bet 1/4 | c1=0)": p_bet_1_4_given_c1_0,
        "P(bet 3/4 | c1=1)": p_bet_3_4_given_c1_1,
    }


def extract_wallet_counts(counts):
    wallet_counts = {state: 0 for state in PAYOFF_BY_WALLET_STATE}

    for bitstring, shots in counts.items():
        cleaned = clean_bitstring(bitstring)
        if len(cleaned) < 8 or cleaned[-1] == "1":
            continue
        wallet = cleaned[0:2]
        if wallet not in wallet_counts:
            raise ValueError(f"Unexpected wallet state '{wallet}' from bitstring '{bitstring}'")
        wallet_counts[wallet] += shots

    return wallet_counts


def extract_guessing_accuracy(counts):
    correct_shots = 0
    total_shots = 0

    for bitstring, shots in counts.items():
        cleaned = clean_bitstring(bitstring)
        if len(cleaned) < 7:
            raise ValueError(f"Bitstring too short to contain G and M2 bits: {bitstring}")

        # Exclude undo branch: Alice chose A2.
        if cleaned[-1] == "1":
            continue

        guess_bit = cleaned[GUESSING_G_INDEX_FROM_LEFT]
        event_bit = cleaned[GUESSING_M2_INDEX_FROM_LEFT]

        total_shots += shots
        if guess_bit == event_bit:
            correct_shots += shots

    return (correct_shots / total_shots) if total_shots else 0.0


def extract_reflex_accuracy(counts):
    correct_shots = 0
    total_shots = 0

    for bitstring, shots in counts.items():
        cleaned = clean_bitstring(bitstring)
        if len(cleaned) < 6:
            raise ValueError(f"Bitstring too short to contain L and M bits: {bitstring}")

        # Exclude undo branch: Alice chose A2.
        if cleaned[-1] == "1":
            continue

        reflex_bit = cleaned[REFLEX_L_INDEX_FROM_LEFT]
        memory_bit = cleaned[REFLEX_M_INDEX_FROM_LEFT]

        total_shots += shots
        if reflex_bit == memory_bit:
            correct_shots += shots

    return (correct_shots / total_shots) if total_shots else 0.0


def expected_payoff_from_wallet_counts(wallet_counts):
    total_shots = sum(wallet_counts.values())
    if total_shots == 0:
        return 0.0
    total_payoff = sum(PAYOFF_BY_WALLET_STATE[state] * count for state, count in wallet_counts.items())
    return total_payoff / total_shots


def theoretical_expected_payoff(win_probability, stake):
    """Expected net payoff for a fair-odds bet paying 1 on success and costing `stake` to enter."""
    return win_probability - stake


def theory_payoff_for_policy(policy_name):
    if policy_name == "betting":
        per_c1 = {
            "0": theoretical_expected_payoff(1 / 4, 1 / 4),
            "1": theoretical_expected_payoff(3 / 4, 3 / 4),
        }
    elif policy_name == "always_small":
        per_c1 = {
            "0": theoretical_expected_payoff(1 / 4, 1 / 4),
            "1": theoretical_expected_payoff(3 / 4, 1 / 4),
        }
    elif policy_name == "always_large":
        per_c1 = {
            "0": theoretical_expected_payoff(1 / 4, 3 / 4),
            "1": theoretical_expected_payoff(3 / 4, 3 / 4),
        }
    elif policy_name == "random":
        per_c1 = {
            "0": 0.5 * theoretical_expected_payoff(1 / 4, 1 / 4) + 0.5 * theoretical_expected_payoff(1 / 4, 3 / 4),
            "1": 0.5 * theoretical_expected_payoff(3 / 4, 1 / 4) + 0.5 * theoretical_expected_payoff(3 / 4, 3 / 4),
        }
    elif policy_name == "opposite":
        per_c1 = {
            "0": theoretical_expected_payoff(1 / 4, 3 / 4),
            "1": theoretical_expected_payoff(3 / 4, 1 / 4),
        }
    else:
        raise ValueError(f"Unknown theoretical policy: {policy_name}")

    return sum(THEORY_C1_PROBABILITIES[c1] * payoff for c1, payoff in per_c1.items())


def theory_strategy_probabilities(policy_name):
    if policy_name == "born_rule":
        return {
            "P(bet 1/4 | c1=0)": 1.0,
            "P(bet 3/4 | c1=1)": 1.0,
        }
    if policy_name == "random":
        return {
            "P(bet 1/4 | c1=0)": 0.5,
            "P(bet 3/4 | c1=1)": 0.5,
        }
    if policy_name == "opposite":
        return {
            "P(bet 1/4 | c1=0)": 0.0,
            "P(bet 3/4 | c1=1)": 0.0,
        }
    if policy_name == "always_small":
        return {
            "P(bet 1/4 | c1=0)": 1.0,
            "P(bet 3/4 | c1=1)": 0.0,
        }
    if policy_name == "always_large":
        return {
            "P(bet 1/4 | c1=0)": 0.0,
            "P(bet 3/4 | c1=1)": 1.0,
        }
    raise ValueError(f"Unknown theoretical policy: {policy_name}")


def load_backend_result(label: str, data_dir: Path, result_filename: str, run_name: Optional[str] = None):
    run_dir, selection_mode = resolve_run_dir(data_dir, result_filename, run_name)
    result_path = run_dir / result_filename

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = data["agents"]["Betting Agent"]["counts"]
    wallet_counts = extract_wallet_counts(counts)
    guessing_counts = data["agents"]["Guessing Agent"]["counts"]
    reflex_counts = data["agents"]["Reflex Agent"]["counts"]
    return {
        "label": label,
        "run_dir": run_dir,
        "run_name": run_dir.name,
        "selection_mode": selection_mode,
        "source_result_path": result_path.resolve(),
        "strategy_probabilities": extract_strategy_probabilities(counts),
        "observed_payoff": expected_payoff_from_wallet_counts(wallet_counts),
        "guessing_accuracy": extract_guessing_accuracy(guessing_counts),
        "reflex_accuracy": extract_reflex_accuracy(reflex_counts),
    }


def build_output_dir(label: Optional[str] = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = None
    if label:
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")
    folder_name = f"{timestamp}__{safe_label}" if safe_label else f"{timestamp}__strategy_comparison"
    return PLOTS_ROOT / folder_name


def plot_strategy_comparison(results, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = [r"$P(\mathrm{bet}\;1/4\mid c_1=0)$", r"$P(\mathrm{bet}\;3/4\mid c_1=1)$"]
    x = np.arange(len(categories))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    zero_marker_height = 0.008

    for idx, result in enumerate(results):
        values = [
            result["strategy_probabilities"]["P(bet 1/4 | c1=0)"],
            result["strategy_probabilities"]["P(bet 3/4 | c1=1)"],
        ]
        offset = (idx - (len(results) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=result["label"],
        )
        for bar, value in zip(bars, values):
            if np.isclose(value, 0.0):
                ax.hlines(
                    y=zero_marker_height,
                    xmin=bar.get_x(),
                    xmax=bar.get_x() + bar.get_width(),
                    colors=bar.get_facecolor(),
                    linewidth=3.0,
                )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.015, 0.965),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.axhline(
        1.0,
        color=IDEAL_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Ideal Born-rule agent",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Born-Rule Strategy Comparison")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.86, 1))

    plot_path = output_dir / "betting_agent_ideal_strategy_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_theory_strategy_comparison(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = [r"$P(\mathrm{bet}\;1/4\mid c_1=0)$", r"$P(\mathrm{bet}\;3/4\mid c_1=1)$"]
    x = np.arange(len(categories))
    width = 0.14

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    zero_marker_height = 0.008

    comparison_series = [
        ("Born-rule", theory_strategy_probabilities("born_rule")),
        ("Random", theory_strategy_probabilities("random")),
        ("Opposite", theory_strategy_probabilities("opposite")),
        ("Always 1/4", theory_strategy_probabilities("always_small")),
        ("Always 3/4", theory_strategy_probabilities("always_large")),
    ]

    for idx, (label, probabilities) in enumerate(comparison_series):
        values = [
            probabilities["P(bet 1/4 | c1=0)"],
            probabilities["P(bet 3/4 | c1=1)"],
        ]
        offset = (idx - (len(comparison_series) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
        )
        for bar, value in zip(bars, values):
            if np.isclose(value, 0.0):
                ax.hlines(
                    y=zero_marker_height,
                    xmin=bar.get_x(),
                    xmax=bar.get_x() + bar.get_width(),
                    colors=bar.get_facecolor(),
                    linewidth=3.0,
                )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.015, 0.965),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Theory Strategy Comparison")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.84, 1))

    plot_path = output_dir / "betting_agent_theory_strategy_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_payoff_comparison(results, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        "Betting\n(noiseless)",
        "Betting\n(fake)",
        "Betting\n(real)",
        "Random",
        "Opposite",
        "Always 1/4",
        "Always 3/4",
    ]
    values = [
        next(result["observed_payoff"] for result in results if result["label"] == "Noiseless"),
        next(result["observed_payoff"] for result in results if result["label"] == "Fake hardware"),
        next(result["observed_payoff"] for result in results if result["label"] == "Real hardware"),
        theory_payoff_for_policy("random"),
        theory_payoff_for_policy("opposite"),
        theory_payoff_for_policy("always_small"),
        theory_payoff_for_policy("always_large"),
    ]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.bar(x, values)
    zero_marker_height = 0.008
    for bar, value in zip(bars, values):
        if np.isclose(value, 0.0):
            ax.hlines(
                y=zero_marker_height,
                xmin=bar.get_x(),
                xmax=bar.get_x() + bar.get_width(),
                colors=bar.get_facecolor(),
                linewidth=3.0,
            )
        text_y = value + 0.015 if value >= 0 else value - 0.055
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{value:.3f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    ax.axhline(0.0, color=IDEAL_COLOR, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Expected payoff")
    ax.set_ylim(-0.33, 0.30)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Expected Payoff Comparison")
    fig.tight_layout()

    plot_path = output_dir / "betting_agent_payoff_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_guessing_accuracy(results, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Noiseless", "Fake hardware", "Real hardware"]
    values = [
        next(result["guessing_accuracy"] for result in results if result["label"] == "Noiseless"),
        next(result["guessing_accuracy"] for result in results if result["label"] == "Fake hardware"),
        next(result["guessing_accuracy"] for result in results if result["label"] == "Real hardware"),
    ]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.bar(x, values)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.015, 0.985),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(
        0.75,
        color=IDEAL_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Ideal accuracy = 0.75",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Guess accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Guessing Agent Accuracy")
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = output_dir / "guessing_agent_accuracy_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_reflex_accuracy(results, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Noiseless", "Fake hardware", "Real hardware"]
    values = [
        next(result["reflex_accuracy"] for result in results if result["label"] == "Noiseless"),
        next(result["reflex_accuracy"] for result in results if result["label"] == "Fake hardware"),
        next(result["reflex_accuracy"] for result in results if result["label"] == "Real hardware"),
    ]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.bar(x, values)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.015, 0.985),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(
        1.0,
        color=IDEAL_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Ideal accuracy = 1.0",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reflex accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Reflex Agent Accuracy")
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = output_dir / "reflex_agent_accuracy_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def print_payoff_summary(results):
    print("\nExpected payoff comparison:")
    for result in results:
        print(f"  Betting agent ({result['label']}): {result['observed_payoff']:.4f}")
    print(f"  Random agent (theory): {theory_payoff_for_policy('random'):.4f}")
    print(f"  Opposite agent (theory): {theory_payoff_for_policy('opposite'):.4f}")
    print(f"  Always-1/4 agent (theory): {theory_payoff_for_policy('always_small'):.4f}")
    print(f"  Always-3/4 agent (theory): {theory_payoff_for_policy('always_large'):.4f}")


def print_guessing_summary(results):
    print("\nGuessing agent accuracy:")
    for result in results:
        print(f"  {result['label']}: {result['guessing_accuracy']:.4f}")


def print_reflex_summary(results):
    print("\nReflex agent accuracy:")
    for result in results:
        print(f"  {result['label']}: {result['reflex_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Create the Betting Agent Born-rule strategy comparison plot."
    )
    parser.add_argument("--noiseless-run", type=str, default=None, help="Run folder name inside data/data_noiseless_simulation.")
    parser.add_argument("--fake-run", type=str, default=None, help="Run folder name inside data/data_fake_hardware.")
    parser.add_argument("--real-run", type=str, default=None, help="Run folder name inside data/data_real_hardware.")
    parser.add_argument("--label", type=str, default=None, help="Optional short label to include in the output folder name.")
    args = parser.parse_args()

    results = [
        load_backend_result("Noiseless", DATA_DIR_NOISELESS, "noiseless_simulation.json", run_name=args.noiseless_run),
        load_backend_result("Fake hardware", DATA_DIR_FAKE, "fake_hardware_noise_sim.json", run_name=args.fake_run),
        load_backend_result("Real hardware", DATA_DIR_REAL, "real_hardware_run.json", run_name=args.real_run),
    ]

    output_dir = build_output_dir(args.label)
    strategy_plot_path = plot_strategy_comparison(results, output_dir)
    theory_strategy_plot_path = plot_theory_strategy_comparison(output_dir)
    payoff_plot_path = plot_payoff_comparison(results, output_dir)
    guessing_plot_path = plot_guessing_accuracy(results, output_dir)
    reflex_plot_path = plot_reflex_accuracy(results, output_dir)
    print_payoff_summary(results)
    print_guessing_summary(results)
    print_reflex_summary(results)
    print(f"Saved strategy comparison plot to: {strategy_plot_path}")
    print(f"Saved theory strategy comparison plot to: {theory_strategy_plot_path}")
    print(f"Saved payoff comparison plot to: {payoff_plot_path}")
    print(f"Saved guessing accuracy plot to: {guessing_plot_path}")
    print(f"Saved reflex accuracy plot to: {reflex_plot_path}")


if __name__ == "__main__":
    main()
