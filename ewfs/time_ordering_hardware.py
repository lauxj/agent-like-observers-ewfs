from pathlib import Path
import json
import argparse
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def save_visualizations_for_agent(viz_dir: Path, agent_name: str, metadata: dict):
    rows = parse_scheduler_timing_rows(metadata)
    if not rows:
        print(f"No scheduler timing rows found for {agent_name}.")
        return

    df = pd.DataFrame(rows).sort_values(["start", "resource", "op"])
    summary = summarize_rows(df)
    ordering = get_ordering_rows(df)
    important = get_important_rows(df)
    safe_name = agent_name.lower().replace(" ", "_")

    summary.to_csv(viz_dir / f"{safe_name}_full_timing.csv", index=False)
    important.to_csv(viz_dir / f"{safe_name}_important_timing.csv", index=False)

    print(
        f"{agent_name}: saved {len(summary)} full rows and {len(important)} important rows to {viz_dir}"
    )


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
    metadata_path = run_dir / "scheduler_timing_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find {metadata_path}")

    timing_data = load_json(metadata_path)
    viz_dir = run_dir / "scheduler_timing_tables"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for agent_name, metadata in timing_data.items():
        save_visualizations_for_agent(viz_dir, agent_name, metadata)

    print(f"Done. Timing tables saved in: {viz_dir}")


if __name__ == "__main__":
    main()