import argparse
from pathlib import Path
import pandas as pd


def find_best_consecutive_block(csv_path: str, block_size: int = 5, error_column: str = "Readout assignment error"):
    df = pd.read_csv(csv_path)

    required = {"Qubit", error_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    work = df[["Qubit", error_column]].copy()
    work = work.dropna()
    work["Qubit"] = work["Qubit"].astype(int)
    work[error_column] = pd.to_numeric(work[error_column], errors="raise")
    work = work.sort_values("Qubit").reset_index(drop=True)

    best = None

    for start in range(len(work) - block_size + 1):
        block = work.iloc[start:start + block_size]
        qubits = block["Qubit"].to_list()

        # Require consecutive qubit numbers, e.g. 44,45,46,47,48
        if qubits != list(range(qubits[0], qubits[0] + block_size)):
            continue

        total_error = block[error_column].sum()

        if best is None or total_error < best["total_error"]:
            best = {
                "start_qubit": qubits[0],
                "end_qubit": qubits[-1],
                "qubits": qubits,
                "total_error": total_error,
                "block": block,
            }

    if best is None:
        raise RuntimeError(
            f"No consecutive block of size {block_size} was found in the dataset."
        )

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the consecutive block of qubits with the lowest summed readout error."
    )
    parser.add_argument("csv_path", help="Path to the calibration CSV file")
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Number of consecutive qubits in the block (default: 5)",
    )
    parser.add_argument(
        "--error-column",
        default="Readout assignment error",
        help='Column to minimize (default: "Readout assignment error")',
    )
    args = parser.parse_args()

    result = find_best_consecutive_block(
        csv_path=args.csv_path,
        block_size=args.block_size,
        error_column=args.error_column,
    )

    print(
        f"Best consecutive block of {args.block_size} qubits: "
        f"Q{result['start_qubit']}-Q{result['end_qubit']}"
    )
    print(f"Summed {args.error_column}: {result['total_error']:.8f}\n")
    print("Individual values:")
    for _, row in result["block"].iterrows():
        print(f"Q{int(row['Qubit'])}: {row[args.error_column]:.8f}")
