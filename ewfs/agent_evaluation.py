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

LF_AGENT_NAMES = ["Betting Agent", "Guessing Agent", "Reflex Agent"]
LF_TERM_SPECS = [
    ("E11", -1.0, r"$-\langle A_1 B_1 \rangle$"),
    ("E12", 1.0, r"$\langle A_1 B_2 \rangle$"),
    ("E21", -1.0, r"$-\langle A_2 B_1 \rangle$"),
    ("E22", -1.0, r"$-\langle A_2 B_2 \rangle$"),
]
LF_TERM_COLORS = ["#FFF3BF", "#5C7CFA", "#D0EBFF", "#9C36B5"]
LF_CORRELATOR_LABELS = [
    r"$\langle A_1 B_1 \rangle$",
    r"$\langle A_1 B_2 \rangle$",
    r"$\langle A_2 B_1 \rangle$",
    r"$\langle A_2 B_2 \rangle$",
]
LF_ANALYTIC_CORRELATORS = {
    "E11": -1.0 / np.sqrt(2.0),
    "E12": 1.0 / np.sqrt(2.0),
    "E21": -1.0 / np.sqrt(2.0),
    "E22": -1.0 / np.sqrt(2.0),
}
BACKEND_COLORS = {
    "Noiseless": "#1F77B4",
    "Fake hardware": "#FF7F0E",
    "Real hardware": "#2CA02C",
}
BACKEND_LABELS = ["Noiseless", "Fake hardware", "Real hardware"]
THEORY_COMPARISON_COLORS = {
    "Born-rule": "#9467BD",
    "Random": "#7F7F7F",
    "Opposite": "#8C564B",
    "Always 1/4": "#17BECF",
    "Always 3/4": "#BCBD22",
}
PAYOFF_COLORS = [
    BACKEND_COLORS["Noiseless"],
    BACKEND_COLORS["Fake hardware"],
    BACKEND_COLORS["Real hardware"],
    THEORY_COMPARISON_COLORS["Random"],
    THEORY_COMPARISON_COLORS["Opposite"],
    THEORY_COMPARISON_COLORS["Always 1/4"],
    THEORY_COMPARISON_COLORS["Always 3/4"],
]


def style_bar_axes(ax, title: str, ylabel: str):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def clean_bitstring(bitstring: str) -> str:
    return "".join(ch for ch in str(bitstring) if ch in {"0", "1"})


def pm(bit: str) -> int:
    return 1 if bit == "0" else -1


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


def result_value(results, label: str, key: str):
    return next(result[key] for result in results if result["label"] == label)


def backend_values(results, key: str):
    return [result_value(results, label, key) for label in BACKEND_LABELS]


def draw_zero_marker(ax, bar, color, height: float = 0.008):
    ax.hlines(
        y=height,
        xmin=bar.get_x(),
        xmax=bar.get_x() + bar.get_width(),
        colors=color,
        linewidth=3.0,
    )


def annotate_vertical_bars(ax, bars, values, *, errors=None, upper_cap: Optional[float] = None):
    if errors is None:
        errors = [0.0] * len(values)

    for bar, value, error in zip(bars, values, errors):
        if np.isclose(value, 0.0):
            draw_zero_marker(ax, bar, bar.get_facecolor())

        if value >= 0:
            text_y = value + error + 0.015
            va = "bottom"
        else:
            text_y = value - error - 0.055
            va = "top"

        if upper_cap is not None:
            text_y = min(text_y, upper_cap)

        label = f"{value:.3f}" if np.isclose(error, 0.0) else f"{value:.3f}\n± {error:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            label,
            ha="center",
            va=va,
            fontsize=9,
        )


def extract_strategy_probabilities(counts):
    stats = {
        "0": {"shots": 0, "correct_bets": 0},
        "1": {"shots": 0, "correct_bets": 0},
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
            stats[c1]["correct_bets"] += shots
        if c1 == "1" and np.isclose(stake, 3 / 4):
            stats[c1]["correct_bets"] += shots

    p_bet_1_4_given_c1_0 = stats["0"]["correct_bets"] / stats["0"]["shots"] if stats["0"]["shots"] else 0.0
    p_bet_3_4_given_c1_1 = stats["1"]["correct_bets"] / stats["1"]["shots"] if stats["1"]["shots"] else 0.0
    stderr_c1_0 = np.sqrt(p_bet_1_4_given_c1_0 * (1.0 - p_bet_1_4_given_c1_0) / stats["0"]["shots"]) if stats["0"]["shots"] else 0.0
    stderr_c1_1 = np.sqrt(p_bet_3_4_given_c1_1 * (1.0 - p_bet_3_4_given_c1_1) / stats["1"]["shots"]) if stats["1"]["shots"] else 0.0

    return {
        "P(bet 1/4 | c1=0)": p_bet_1_4_given_c1_0,
        "P(bet 3/4 | c1=1)": p_bet_3_4_given_c1_1,
        "P(bet 1/4 | c1=0) stderr": stderr_c1_0,
        "P(bet 3/4 | c1=1) stderr": stderr_c1_1,
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


def extract_binary_accuracy(counts, *, min_length: int, first_index: int, second_index: int, bit_names: str):
    correct_shots = 0
    total_shots = 0

    for bitstring, shots in counts.items():
        cleaned = clean_bitstring(bitstring)
        if len(cleaned) < min_length:
            raise ValueError(f"Bitstring too short to contain {bit_names}: {bitstring}")

        # Exclude undo branch: Alice chose A2.
        if cleaned[-1] == "1":
            continue

        total_shots += shots
        if cleaned[first_index] == cleaned[second_index]:
            correct_shots += shots

    accuracy = (correct_shots / total_shots) if total_shots else 0.0
    stderr = np.sqrt(accuracy * (1.0 - accuracy) / total_shots) if total_shots else 0.0
    return {
        "accuracy": accuracy,
        "stderr": stderr,
        "correct_shots": correct_shots,
        "total_shots": total_shots,
    }


def extract_guessing_accuracy(counts):
    return extract_binary_accuracy(
        counts,
        min_length=7,
        first_index=GUESSING_G_INDEX_FROM_LEFT,
        second_index=GUESSING_M2_INDEX_FROM_LEFT,
        bit_names="G and M bits",
    )


def extract_reflex_accuracy(counts):
    return extract_binary_accuracy(
        counts,
        min_length=6,
        first_index=REFLEX_L_INDEX_FROM_LEFT,
        second_index=REFLEX_M_INDEX_FROM_LEFT,
        bit_names="L and M bits",
    )


def expected_payoff_from_wallet_counts(wallet_counts):
    total_shots = sum(wallet_counts.values())
    if total_shots == 0:
        return 0.0
    total_payoff = sum(PAYOFF_BY_WALLET_STATE[state] * count for state, count in wallet_counts.items())
    return total_payoff / total_shots


def expected_payoff_stderr_from_wallet_counts(wallet_counts):
    total_shots = sum(wallet_counts.values())
    if total_shots == 0:
        return 0.0

    payoff_values = np.array([PAYOFF_BY_WALLET_STATE[state] for state in wallet_counts])
    payoff_counts = np.array([wallet_counts[state] for state in wallet_counts])
    probabilities = payoff_counts / total_shots
    mean_payoff = np.sum(probabilities * payoff_values)
    variance = np.sum(probabilities * (payoff_values - mean_payoff) ** 2)
    return np.sqrt(variance / total_shots)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lf_violations_for_run(run_dir: Path):
    lf_path = run_dir / "lf_violations" / "lf_violations.json"
    if not lf_path.exists():
        raise FileNotFoundError(f"LF violations file not found: {lf_path.resolve()}")
    return load_json(lf_path)


def lf_correlator_series_from_saved_results(agent_lf_data):
    correlators = agent_lf_data["correlators"]
    estimated_shots_per_setting = agent_lf_data["total_shots"] / 4.0
    return {
        key: {
            "value": correlators[key],
            "stderr": np.sqrt(max(0.0, 1.0 - correlators[key] ** 2) / estimated_shots_per_setting),
        }
        for key in ["E11", "E12", "E21", "E22"]
    }


def agent_label_to_filename(agent_name: str) -> str:
    return agent_name.lower().replace(" ", "_")


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


def load_backend_result(label: str, data_dir: Path, result_filename: str, run_name: Optional[str] = None):
    run_dir, selection_mode = resolve_run_dir(data_dir, result_filename, run_name)
    result_path = run_dir / result_filename

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = data["agents"]["Betting Agent"]["counts"]
    wallet_counts = extract_wallet_counts(counts)
    guessing_counts = data["agents"]["Guessing Agent"]["counts"]
    reflex_counts = data["agents"]["Reflex Agent"]["counts"]
    guessing_stats = extract_guessing_accuracy(guessing_counts)
    reflex_stats = extract_reflex_accuracy(reflex_counts)
    return {
        "label": label,
        "run_dir": run_dir,
        "run_name": run_dir.name,
        "selection_mode": selection_mode,
        "source_result_path": result_path.resolve(),
        "strategy_probabilities": extract_strategy_probabilities(counts),
        "observed_payoff": expected_payoff_from_wallet_counts(wallet_counts),
        "observed_payoff_stderr": expected_payoff_stderr_from_wallet_counts(wallet_counts),
        "guessing_accuracy": guessing_stats["accuracy"],
        "guessing_accuracy_stderr": guessing_stats["stderr"],
        "guessing_accuracy_shots": guessing_stats["total_shots"],
        "reflex_accuracy": reflex_stats["accuracy"],
        "reflex_accuracy_stderr": reflex_stats["stderr"],
        "reflex_accuracy_shots": reflex_stats["total_shots"],
    }


def build_output_dir(label: Optional[str] = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = None
    if label:
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")
    folder_name = f"{timestamp}__{safe_label}" if safe_label else f"{timestamp}__strategy_comparison"
    return PLOTS_ROOT / folder_name


def plot_born_rule_accuracy(results, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = [r"$P(\mathrm{bet}\;1/4\mid c_1=0)$", r"$P(\mathrm{bet}\;3/4\mid c_1=1)$"]
    x = np.arange(len(categories))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for idx, result in enumerate(results):
        values = [
            result["strategy_probabilities"]["P(bet 1/4 | c1=0)"],
            result["strategy_probabilities"]["P(bet 3/4 | c1=1)"],
        ]
        errors = [
            result["strategy_probabilities"]["P(bet 1/4 | c1=0) stderr"],
            result["strategy_probabilities"]["P(bet 3/4 | c1=1) stderr"],
        ]
        offset = (idx - (len(results) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            yerr=errors,
            capsize=5,
            ecolor="#333333",
            width=width,
            label=result["label"],
            color=BACKEND_COLORS[result["label"]],
            edgecolor="black",
            linewidth=1.0,
        )
        annotate_vertical_bars(ax, bars, values, errors=errors, upper_cap=0.965)

    ax.axhline(
        1.0,
        color=IDEAL_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Ideal Born-rule agent",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.0, 1.05)
    style_bar_axes(ax, "Born-Rule Agent Accuracy", "Accuracy")
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    fig.tight_layout()

    plot_path = output_dir / "born_rule_agent_accuracy_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_always_large_vs_betting_payoff_comparison(results, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        "Born-rule\n(noiseless)",
        "Born-rule\n(fake)",
        "Born-rule\n(real)",
        "Always 3/4",
    ]
    values = [*backend_values(results, "observed_payoff"), theory_payoff_for_policy("always_large")]
    errors = [*backend_values(results, "observed_payoff_stderr"), 0.0]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        ecolor="#333333",
        color=[
            BACKEND_COLORS["Noiseless"],
            BACKEND_COLORS["Fake hardware"],
            BACKEND_COLORS["Real hardware"],
            THEORY_COMPARISON_COLORS["Always 3/4"],
        ],
        edgecolor="black",
        linewidth=1.0,
    )
    annotate_vertical_bars(ax, bars, values, errors=errors)

    ax.axhline(0.0, color=IDEAL_COLOR, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.33, 0.30)
    style_bar_axes(ax, "Born-Rule Agent vs Always-3/4 Payoff", "Expected payoff")
    fig.tight_layout()

    plot_path = output_dir / "betting_agent_vs_always_3_4_payoff_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_real_hardware_lf_correlator_comparisons(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    real_run_dir = result_value(results, "Real hardware", "run_dir")
    real_lf_results = load_lf_violations_for_run(real_run_dir)

    saved_paths = []
    classical_bound = 2.0
    tsirelson_bound = 2.0 * np.sqrt(2.0)

    for agent_name in LF_AGENT_NAMES:
        real_series = lf_correlator_series_from_saved_results(real_lf_results["agents"][agent_name])

        raw_values = np.array([real_series[key]["value"] for key, _, _ in LF_TERM_SPECS])
        raw_errors = np.array([2.0 * real_series[key]["stderr"] for key, _, _ in LF_TERM_SPECS])
        raw_theory_values = np.array([LF_ANALYTIC_CORRELATORS[key] for key, _, _ in LF_TERM_SPECS])
        signed_term_values = np.array([sign * real_series[key]["value"] for key, sign, _ in LF_TERM_SPECS])
        signed_term_errors = np.array([2.0 * real_series[key]["stderr"] for key, _, _ in LF_TERM_SPECS])
        signed_term_theory_values = np.array([sign * LF_ANALYTIC_CORRELATORS[key] for key, sign, _ in LF_TERM_SPECS])
        signed_term_labels = [label for _, _, label in LF_TERM_SPECS]

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.15},
        )

        x_pos = np.arange(len(LF_TERM_SPECS))
        bar_width = 0.8

        for idx in range(len(LF_TERM_SPECS)):
            ax1.bar(
                x_pos[idx],
                raw_values[idx],
                width=bar_width,
                color=LF_TERM_COLORS[idx],
                edgecolor="black",
                yerr=raw_errors[idx],
                capsize=5,
                zorder=2,
            )
            ax1.bar(
                x_pos[idx],
                raw_theory_values[idx],
                width=bar_width,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=2,
                zorder=3,
            )
            positive_reference = max(raw_values[idx] + raw_errors[idx], raw_theory_values[idx])
            negative_reference = min(raw_values[idx] - raw_errors[idx], raw_theory_values[idx])
            value_text_y = positive_reference + 0.08 if raw_values[idx] >= 0 else negative_reference - 0.08
            value_text_va = "bottom" if raw_values[idx] >= 0 else "top"
            ax1.text(
                x_pos[idx],
                value_text_y,
                f"{raw_values[idx]:.3f}\n± {raw_errors[idx]:.3f}",
                ha="center",
                va=value_text_va,
                fontsize=9,
                zorder=5,
            )

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(LF_CORRELATOR_LABELS, fontsize=12)
        ax1.set_xlim(-0.8, len(LF_TERM_SPECS) - 0.2)
        ax1.set_ylim(-1.05, 1.05)
        ax1.set_ylabel("Correlator value")
        ax1.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax1.grid(axis="y", alpha=0.25)
        ax1.set_title(f"{agent_name}: Real-Hardware Raw LF Correlators")
        ax1.plot([], [], color="red", linestyle="--", linewidth=2, label="Theoretical quantum maximum")
        ax1.plot([], [], color="black", linewidth=1.5, marker="|", markersize=10, label=r"$2\sigma$ uncertainty")
        ax1.legend(loc="upper right", fontsize=10, frameon=True)

        left_exp = 0.0
        left_th = 0.0
        bar_height = 0.5
        cumulative_centers = []
        cumulative_errors = []
        cumulative_variance = 0.0
        segment_label_y = bar_height / 2 + 0.18

        for idx, label in enumerate(signed_term_labels):
            width_exp = abs(signed_term_values[idx])
            width_th = abs(signed_term_theory_values[idx])
            term_two_sigma = signed_term_errors[idx]

            ax2.barh(
                0,
                width_exp,
                height=bar_height,
                left=left_exp,
                color=LF_TERM_COLORS[idx],
                edgecolor="none",
                zorder=2,
            )
            ax2.barh(
                0,
                width_th,
                height=bar_height + 0.1,
                left=left_th,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=1.5,
                zorder=3,
            )
            ax2.text(
                left_exp + (width_exp / 2),
                segment_label_y,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
            )
            left_exp += width_exp
            left_th += width_th

            term_sigma = term_two_sigma / 2.0
            cumulative_variance += term_sigma ** 2
            cumulative_centers.append(left_exp)
            cumulative_errors.append(2.0 * np.sqrt(cumulative_variance))

        ax2.errorbar(
            cumulative_centers,
            [0] * len(cumulative_centers),
            xerr=[err / 2.0 for err in cumulative_errors],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            zorder=4,
        )

        threshold_ymin = 0.0
        threshold_ymax = 0.92
        for threshold in [classical_bound, tsirelson_bound]:
            ax2.axvline(
                x=threshold,
                color="red",
                linewidth=2.2,
                ymin=threshold_ymin,
                ymax=threshold_ymax,
                zorder=5,
            )

        final_two_sigma = cumulative_errors[-1]
        final_violation = float(np.sum(signed_term_values)) - classical_bound
        final_text_x = cumulative_centers[-1] + final_two_sigma / 2.0 + 0.08
        ax2.text(
            final_text_x,
            0.0,
            f"S = {final_violation:.3f}\n± {final_two_sigma:.3f}",
            ha="left",
            va="center",
            fontsize=10,
        )

        right_limit = max(tsirelson_bound + 0.12, final_text_x + 0.42)
        ax2.set_xlim(0, right_limit)
        ax2.set_ylim(-1, 1)
        ax2.set_yticks([])
        for spine in ["top", "left", "right"]:
            ax2.spines[spine].set_visible(False)

        ax2.set_xticks([0, classical_bound, tsirelson_bound])
        ax2.set_xticklabels(["0", "2", r"$2\sqrt{2}$"], fontsize=12)
        ax2.spines["bottom"].set_linewidth(1.5)

        fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.12, hspace=0.18)

        plot_path = output_dir / f"real_hardware_{agent_label_to_filename(agent_name)}_lf_correlator_comparison.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


def plot_accuracy_comparison(
    results,
    output_dir: Path,
    *,
    value_key: str,
    error_key: str,
    title: str,
    ylabel: str,
    ideal_value: float,
    ideal_label: str,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    values = backend_values(results, value_key)
    errors = backend_values(results, error_key)
    x = np.arange(len(BACKEND_LABELS))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        ecolor="#333333",
        color=[BACKEND_COLORS[label] for label in BACKEND_LABELS],
        edgecolor="black",
        linewidth=1.0,
    )
    annotate_vertical_bars(ax, bars, values, errors=errors, upper_cap=0.985)

    ax.axhline(
        ideal_value,
        color=IDEAL_COLOR,
        linestyle="--",
        linewidth=1.8,
        label=ideal_label,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(BACKEND_LABELS)
    ax.set_ylim(0.0, 1.05)
    style_bar_axes(ax, title, ylabel)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    fig.tight_layout()

    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_guessing_accuracy(results, output_dir: Path) -> Path:
    return plot_accuracy_comparison(
        results,
        output_dir,
        value_key="guessing_accuracy",
        error_key="guessing_accuracy_stderr",
        title="Guessing Agent Accuracy",
        ylabel="Guess accuracy",
        ideal_value=0.75,
        ideal_label="Ideal accuracy = 0.75",
        filename="guessing_agent_accuracy_comparison.png",
    )


def plot_reflex_accuracy(results, output_dir: Path) -> Path:
    return plot_accuracy_comparison(
        results,
        output_dir,
        value_key="reflex_accuracy",
        error_key="reflex_accuracy_stderr",
        title="Reflex Agent Accuracy",
        ylabel="Reflex accuracy",
        ideal_value=1.0,
        ideal_label="Ideal accuracy = 1.0",
        filename="reflex_agent_accuracy_comparison.png",
    )


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
        print(
            f"  {result['label']}: {result['guessing_accuracy']:.4f} "
            f"+/- {result['guessing_accuracy_stderr']:.4f} "
            f"(n={result['guessing_accuracy_shots']})"
        )


def print_reflex_summary(results):
    print("\nReflex agent accuracy:")
    for result in results:
        print(
            f"  {result['label']}: {result['reflex_accuracy']:.4f} "
            f"+/- {result['reflex_accuracy_stderr']:.4f} "
            f"(n={result['reflex_accuracy_shots']})"
        )


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
    born_rule_accuracy_plot_path = plot_born_rule_accuracy(results, output_dir)
    always_large_vs_betting_plot_path = plot_always_large_vs_betting_payoff_comparison(results, output_dir)
    lf_correlator_plot_paths = plot_real_hardware_lf_correlator_comparisons(results, output_dir)
    guessing_plot_path = plot_guessing_accuracy(results, output_dir)
    reflex_plot_path = plot_reflex_accuracy(results, output_dir)
    print_payoff_summary(results)
    print_guessing_summary(results)
    print_reflex_summary(results)
    print(f"Saved Born-rule accuracy plot to: {born_rule_accuracy_plot_path}")
    print(f"Saved betting-vs-always-3/4 payoff comparison plot to: {always_large_vs_betting_plot_path}")
    for plot_path in lf_correlator_plot_paths:
        print(f"Saved real-hardware LF correlator plot to: {plot_path}")
    print(f"Saved guessing accuracy plot to: {guessing_plot_path}")
    print(f"Saved reflex accuracy plot to: {reflex_plot_path}")


if __name__ == "__main__":
    main()
