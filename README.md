# Master's Thesis Project

This repository contains the code and resources for my master's thesis project.

## Repository Layout

- `src/ewfs/circuits/`: circuit builders for the EWFS agents and accuracy-test circuits.
- `src/ewfs/experiments/`: experiment runners for noiseless simulation, fake hardware, real hardware, and IBM transpilation.
- `src/ewfs/analysis/`: post-processing, LF-violation calculations, and plotting utilities.
- `scripts/`: small terminal entrypoints that call into the package.
- `notebooks/`: exploratory notebooks.
- `data/paper_data/`: the selected paper/reproducibility runs used by the evaluation script.
- `data/data_*`: local experiment outputs generated while running new simulations or hardware jobs.
- `results/`: generated plots and analysis outputs.

## Running

From the repository root, run the main experiment pipeline with:

```bash
python scripts/run_experiment.py --shots-main 5000 --exclude-accuracy-tests
```

Generate the main agent-evaluation outputs with:

```bash
python scripts/evaluate_agents.py
```

Generate time-ordering plots for the latest real-hardware run with:

```bash
python scripts/plot_time_ordering.py
```

The project can also be installed in editable mode:

```bash
pip install -e .
```

After that, the same entrypoints are available as `ewfs-run`, `ewfs-evaluate`,
`ewfs-plot-time-ordering`, and `ewfs-plot-connectivity`.
