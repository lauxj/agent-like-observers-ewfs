from pathlib import Path
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_REAL = PROJECT_ROOT / "data" / "data_real_hardware"


INTERESTING_KEYWORDS = [
    "measure",
    "broadcast",
    "receive",
    "cz",
    "cx",
    "barrier",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_run_dir() -> Path:
    run_dirs = [p for p in DATA_DIR_REAL.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {DATA_DIR_REAL}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def parse_scheduler_timing_rows(metadata: dict):
    """Parse IBM scheduler timing metadata into a list of timing rows."""
    meta_root = metadata.get("pub_metadata", metadata)
    timing_str = (
        meta_root.get("compilation", {})
        .get("scheduler_timing", {})
        .get("timing", "")
    )
    if not timing_str:
        return []

    rows = []
    for line in timing_str.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue

        block, op, resource, start, duration, kind = parts
        try:
            start = int(start)
            duration = int(duration)
        except ValueError:
            continue

        rows.append(
            {
                "block": block,
                "op": op,
                "resource": resource,
                "start": start,
                "duration": duration,
                "end": start + duration,
                "kind": kind,
            }
        )

    return rows


def classify_event(row: dict) -> str:
    """Map raw scheduler events to a smaller set of readable event types."""
    op = str(row["op"]).lower()
    kind = str(row["kind"]).lower()
    resource = str(row["resource"])

    if "measure" in op:
        if kind == "play":
            return "measure_start"
        if kind == "capture":
            return "measure_capture"
        return "measure"

    if "broadcast" in op:
        return "broadcast"

    if "receive" in op:
        return "receive"

    if "cz" in op:
        return "cz"

    if "cx" in op:
        return "cx"

    if kind == "barrier" or "barrier" in op:
        return "barrier"

    if "Qubit" in resource:
        return "qubit_op"

    return "other"


def short_label(op: str) -> str:
    """Make long operation names readable inside the plot."""
    op = str(op)
    if op.startswith("measure_"):
        return op.replace("measure_", "m")
    if op.startswith("INIT_"):
        return op.replace("INIT_", "I")
    if op.startswith("cz_"):
        return op.replace("cz_", "cz")
    if op.startswith("cx_"):
        return op.replace("cx_", "cx")
    return op if len(op) <= 12 else op[:12]


def resource_sort_key(resource: str):
    """Match the original visual grouping more closely."""
    resource = str(resource)
    if resource == "Receive":
        return (0, -1)
    if resource.startswith("Qubit "):
        try:
            return (1, -int(resource.split()[1]))
        except (IndexError, ValueError):
            return (1, 10**9)
    if resource == "Hub":
        return (2, -1)
    if resource.startswith("AWGR"):
        try:
            left, right = resource.split("_")
            awg_num = int(left.replace("AWGR", ""))
            qubit_num = int(right)
            return (3, -awg_num, -qubit_num)
        except (ValueError, IndexError):
            return (3, 10**9, 10**9)
    return (4, resource)


def draw_timing_panel(ax, plot_df: pd.DataFrame, y_positions: dict, event_colors: dict,
                      min_visible_width: int, label_width_threshold: int,
                      marker_threshold: int, x_min: int, x_max: int,
                      show_labels: bool, title: str):
    """Draw one timing panel."""
    window_df = plot_df[(plot_df["start"] < x_max) & (plot_df["end"] > x_min)].copy()
    if window_df.empty:
        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1, len(y_positions))
        return

    for _, row in window_df.iterrows():
        event = classify_event(row.to_dict())
        color = event_colors.get(event, "tab:olive")
        y = y_positions[row["resource"]]
        start = int(row["start"])
        duration = max(int(row["duration"]), 1)
        visible_width = max(duration, min_visible_width)

        ax.broken_barh(
            [(start, visible_width)],
            (y - 0.35, 0.7),
            facecolors=color,
            edgecolors="black",
            linewidth=0.5,
            alpha=0.9,
        )

        if show_labels and visible_width >= label_width_threshold:
            ax.text(
                start + visible_width / 2,
                y,
                short_label(row["op"]),
                ha="center",
                va="center",
                fontsize=7,
                clip_on=True,
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, len(y_positions))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_title(title)



def choose_zoom_window(plot_df: pd.DataFrame, window_fraction: float = 0.28):
    """Choose the densest time window for a zoomed panel."""
    max_end = int(plot_df["end"].max()) if not plot_df.empty else 0
    if max_end <= 0:
        return 0, 1

    window_width = max(200, int(window_fraction * max_end))
    step = max(20, window_width // 20)

    best_start = 0
    best_score = -1
    for start in range(0, max(1, max_end - window_width + 1), step):
        end = start + window_width
        mask = (plot_df["start"] < end) & (plot_df["end"] > start)
        score = int(mask.sum())
        if score > best_score:
            best_score = score
            best_start = start

    return best_start, min(max_end, best_start + window_width)


def plot_timing_for_agent(viz_dir: Path, agent_name: str, df: pd.DataFrame):
    """Create an overview plot plus a zoomed timing panel for dense short-gate regions."""
    plot_df = df.copy().sort_values(["start", "resource", "op"]).reset_index(drop=True)
    if plot_df.empty:
        return

    resources = sorted(plot_df["resource"].unique().tolist(), key=resource_sort_key)
    y_positions = {resource: idx for idx, resource in enumerate(resources)}

    event_colors = {
        "measure_start": "tab:red",
        "measure_capture": "tab:orange",
        "broadcast": "tab:purple",
        "receive": "tab:green",
        "cz": "tab:blue",
        "cx": "tab:cyan",
        "barrier": "tab:gray",
        "qubit_op": "tab:brown",
        "measure": "tab:pink",
        "other": "tab:olive",
    }

    max_end = int(plot_df["end"].max()) if not plot_df.empty else 0
    overview_min_visible_width = max(3, int(0.002 * max_end))
    zoom_min_visible_width = max(10, int(0.006 * max_end))
    overview_label_width_threshold = max_end + 1
    zoom_label_width_threshold = max(36, int(0.02 * max_end))
    marker_threshold = max(8, int(0.002 * max_end))

    zoom_start, zoom_end = choose_zoom_window(plot_df)

    fig_height = max(6, 0.55 * len(resources) + 3)
    fig, (ax_overview, ax_zoom) = plt.subplots(
        2, 1,
        figsize=(16, fig_height + 4),
        sharey=True,
        gridspec_kw={"height_ratios": [1.1, 1.4]}
    )

    draw_timing_panel(
        ax_overview,
        plot_df,
        y_positions,
        event_colors,
        overview_min_visible_width,
        overview_label_width_threshold,
        marker_threshold,
        0,
        max_end + max(1, int(0.02 * max_end) + 1),
        show_labels=False,
        title=f"{agent_name} scheduler timing — overview",
    )

    draw_timing_panel(
        ax_zoom,
        plot_df,
        y_positions,
        event_colors,
        zoom_min_visible_width,
        zoom_label_width_threshold,
        marker_threshold,
        zoom_start,
        zoom_end,
        show_labels=True,
        title=f"Dense region zoom: {zoom_start} to {zoom_end}",
    )

    for ax in (ax_overview, ax_zoom):
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(resources)
        ax.set_ylabel("Resource")

    ax_overview.set_xlabel("Scheduler time")
    ax_zoom.set_xlabel("Scheduler time")

    legend_order = [
        "measure_start",
        "measure_capture",
        "broadcast",
        "receive",
        "cz",
        "cx",
        "barrier",
        "qubit_op",
        "measure",
        "other",
    ]
    present_events = []
    for event_name in legend_order:
        if any(classify_event(row.to_dict()) == event_name for _, row in plot_df.iterrows()):
            present_events.append(event_name)

    if present_events:
        handles = [
            Patch(facecolor=event_colors[event_name], edgecolor="black", label=event_name)
            for event_name in present_events
        ]
        handles.append(Patch(facecolor="white", edgecolor="black", label=f"zoom min width ≥ {zoom_min_visible_width}"))
        ax_overview.legend(handles=handles, title="Event type", loc="upper left")

    fig.tight_layout()
    safe_name = safe_label(agent_name)
    fig.savefig(viz_dir / f"{safe_name}_timing_plot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Create a smaller, readable table for checking causal time ordering."""
    summary = df.copy()
    summary["event"] = summary.apply(lambda row: classify_event(row.to_dict()), axis=1)
    summary = summary[
        ["start", "end", "duration", "event", "op", "resource", "kind", "block"]
    ].sort_values(["start", "resource", "op"])
    return summary



def get_ordering_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most relevant rows for understanding temporal ordering."""
    summary = summarize_rows(df)
    ordering = summary[
        summary["event"].isin(
            ["measure_start", "measure_capture", "broadcast", "receive", "cz", "cx", "barrier"]
        )
    ].copy()
    if ordering.empty:
        ordering = summary.copy()
    return ordering.sort_values(["start", "resource", "op"])


def get_important_rows(df: pd.DataFrame) -> pd.DataFrame:
    ordering = get_ordering_rows(df)
    mask = ordering["op"].str.lower().apply(
        lambda s: any(keyword in s for keyword in INTERESTING_KEYWORDS)
    )
    important = ordering[mask].copy()
    if important.empty:
        important = ordering.copy()
    return important.sort_values(["start", "resource", "op"])


def safe_label(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")


def save_visualizations_for_agent(viz_dir: Path, agent_name: str, metadata: dict):
    rows = parse_scheduler_timing_rows(metadata)
    if not rows:
        print(f"No scheduler timing rows found for {agent_name}.")
        return

    df = pd.DataFrame(rows).sort_values(["start", "resource", "op"])
    summary = summarize_rows(df)
    ordering = get_ordering_rows(df)
    important = get_important_rows(df)
    safe_name = safe_label(agent_name)

    summary.to_csv(viz_dir / f"{safe_name}_full_timing.csv", index=False)
    important.to_csv(viz_dir / f"{safe_name}_important_timing.csv", index=False)
    plot_timing_for_agent(viz_dir, agent_name, summary)

    print(
        f"{agent_name}: saved {len(summary)} full rows, {len(important)} important rows, and a timing plot to {viz_dir}"
    )


def save_visualizations_for_run(run_dir: Path):
    metadata_path = run_dir / "scheduler_timing_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find {metadata_path}")

    timing_data = load_json(metadata_path)
    viz_dir = run_dir / "scheduler_timing_tables"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for agent_name, metadata in timing_data.items():
        save_visualizations_for_agent(viz_dir, agent_name, metadata)

    print(f"Done. Timing tables saved in: {viz_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize scheduler timing metadata from a real hardware run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a specific run directory inside data/data_real_hardware. If omitted, the latest run is used.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir()
    save_visualizations_for_run(run_dir)


if __name__ == "__main__":
    main()
