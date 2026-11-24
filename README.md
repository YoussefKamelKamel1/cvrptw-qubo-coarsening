# Graph Coarsening and Adaptive QUBO Penalties for the CVRPTW

Code for solving the Capacitated Vehicle Routing Problem with Time Windows
(CVRPTW) as a Quadratic Unconstrained Binary Optimization (QUBO) problem on a
simulated-annealing backend, with two contributions:

1. **Adaptive penalty calibration.** Uniformly scaling QUBO penalties leaves
   simulated annealing unchanged; the effective lever is the QUBO's internal
   dynamic range, which is dominated by capacity log-slack couplings that are
   non-binding on most small instances. Dropping those and scaling the rest
   reduces raw constraint violations by roughly an order of magnitude at equal
   solver budget.
2. **Learned, tuning-free coarsening.** A GraphSAGE merge scorer replaces the
   hand-tuned spatio-temporal score with a single fixed configuration for all
   instance families. It preserves feasibility better than the per-family-tuned
   heuristic across `N = 10..100`, tuning-free, and keeps the QUBO small enough to
   stay solvable and embeddable at large `N`.

All experiments run on a classical simulated-annealing sampler and OR-Tools
serves as a classical reference. The hardware runs on a D-Wave Advantage2
processor use `scripts/run_qpu.py`; their outputs are in `results/qpu_*.csv`.

## Installation

```bash
python -m pip install -r requirements.txt
```

Install a `torch` build matching your CUDA/CPU before `torch-geometric`. The
GPU is used only for training the GNN; all solving runs on CPU.

## Layout

```
src/          core library (datasets, coarsening, QUBO, solvers, GNN, metrics)
scripts/      experiment drivers and reproduction runners
configs/      YAML configs for the baseline and penalty-ablation runs
data/         Solomon (CVRPTW) and Christofides (CVRP) benchmark instances
results/      trained models, training labels and the result files behind the paper
experiments/  additional experiments, their result files and trained models
figures/      created by the make_*.py scripts
```

## Reproducing the results

```bash
# baseline reproduction and penalty ablation
python scripts/run_phase1.py
python scripts/run_phase2_full.py

# GNN: generate labels, train, evaluate
python scripts/gen_labels.py
python scripts/train_gnn.py
python scripts/run_phase4_full.py

# controls and scaling
python scripts/run_confound.py         # dynamic-range vs. variable-count control
python scripts/run_scorer_ablation.py  # learned vs. fixed merge score
python scripts/run_repair_only.py       # classical repair baseline
python scripts/run_local_search.py      # shared local-search polish
python scripts/run_scaling_large.py     # QUBO size and feasibility vs. N

# robustness
python scripts/run_penalty_larger_n.py  # penalty comparison at N = 20, 40
python scripts/run_sa_schedule.py       # SA temperature range for static penalties
python scripts/run_lofo.py train && python scripts/run_lofo.py eval   # leave-one-family-out
python scripts/compare_architectures.py # GraphSAGE vs. GAT vs. GCN
python scripts/gen_labels_cached.py && python scripts/sweep_reward.py  # reward weights
python scripts/analyze_robustness.py

# figures (written to figures/)
python scripts/make_method_figures.py
python scripts/make_paper_data_figures.py
python scripts/make_confound_figure.py
python scripts/make_phase4_figures.py
python scripts/make_learned_stop_figure.py
python scripts/make_qpu_figures.py
```

The trained coarsening policy is provided at `results/gnn_model_rl.pt`, so the
evaluation and scaling scripts can be run without retraining. The multi-size model
whose rollouts produced its expert-iteration labels (`results/gnn_model_multiN.pt`)
and the training labels
(`results/gnn_labels_*.pt`) are included as well. All result files used for the
figures and tables are in `results/`, so the figure scripts run without
re-running the experiments.

## Additional experiments

`experiments/` holds the runs added in the revision: the Gehring-Homberger and EURO
Meets NeurIPS 2022 benchmarks, the five retrained scorers, the re-tuned and label-free
heuristics, penalty sensitivity, roof duality, route cost and running time up to
N = 100, the improved training, the selection among candidate coarsenings, and the
cost-gap diagnosis. Their result files and trained models are in
`experiments/results/`; each script states at the top what it does and how to run it.

```bash
# coarsener comparison with per-run traces (Solomon, Gehring-Homberger, EURO)
python experiments/review_runner.py --dataset solomon --scoring stored --tiebreak rank --c104 fixed
python experiments/review_runner.py --dataset gh --scoring stored --tiebreak rank
python experiments/review_runner.py --dataset euro --scoring stored --tiebreak rank --arms heuristic_fixed gnn
python experiments/fullcheck_repair.py experiments/results/<tag>/traces.jsonl <dataset> <out.csv>

# additional comparisons (re-tuned and label-free heuristics, retrained models, trunks)
python experiments/retrain_sage.py --arch sage --seeds 0 1 2 3 4
python experiments/rev_runner.py tune
python experiments/rev_runner.py test
python experiments/rev_runner.py nolabel
python experiments/rev_runner.py retrain

# penalty sensitivity, roof duality, OR-Tools reference, cost and time
python experiments/penalty_sensitivity.py
python experiments/reduction_compare.py
python experiments/ortools_ref_test.py
python experiments/analyze_cost_time.py

# improved training, then selection among candidate coarsenings
bash experiments/run_v2.sh
python experiments/v3_data.py && python experiments/v3_train.py
python experiments/v3_test.py --dataset solomon --capped
```

The Gehring-Homberger and EURO instances are not included. Extract the
Gehring-Homberger 200- and 400-customer instance files (`C1_2_1.TXT`, ...) into
`experiments/data_gh/extracted/` and `experiments/data_gh400/extracted/`, and place
the EURO instance files from
github.com/ortec/euro-neurips-vrp-2022-quickstart (CC BY-NC 4.0) in
`experiments/data_euro/raw/`. The lists of EURO instances used are in
`experiments/data_euro/`.

## Solver backend

The default backend is D-Wave's `SimulatedAnnealingSampler` from
`dwave-samplers`. `scripts/run_qpu.py` contains a hardware path (via the D-Wave
Leap framework) with a dry-run mode that validates the pipeline on the simulator;
running it on real hardware requires Leap access.
