# Task order in continual-learning evaluation

Post-training analysis for a study of how task order changes continual-learning comparisons in motion forecasting. The released benchmark contains four tasks, all 24 task permutations, four algorithms, and three training replicates per order–algorithm combination: 288 runs in total.

This repository provides compact evaluation outcomes, reproducible numerical analysis, and result-figure code. The [manuscript repository](https://github.com/lzrbit/Zirui-Li-CL-task-order-influence) contains the paper and its schematic figures. Model training and its configuration are maintained separately; running this analysis does not train a model or load pickle files.

**Project page:** [lzrbit.github.io/task_order](https://lzrbit.github.io/task_order/)

**Manuscript:** *What is the Influence of Task Order in Continual Learning? An Empirical Study and Order-aware Evaluation for Motion Forecasting* — Zirui Li, Yunlong Lin, Haohan Yang, Xiaocong Zhao, and Chen Lv.

## Quick start

Use Python 3.12 for the tested environment:

```sh
git clone git@github.com:lzrbit/task_order.git
cd task_order
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python run.py
```

`run.py` first verifies input hashes, regenerates numerical results and LaTeX tables in `analysis/results/`, and independently checks the main metrics. No original trajectory archive is needed. For compatible dependency ranges instead of the tested versions, use `requirements.txt`.

To generate the numerical figures as PDF and editable SVG:

```sh
python run.py --figures
```

The figure scripts require an installed Arial font, including its bold and italic faces. The numerical analysis does not require Arial. Fonts are not bundled. PNG previews and a machine-readable validation report are optional:

```sh
python run.py --figures --preview-dir previews --report validation_report.json
python run.py --check-only
```

`--check-only` checks the supplied data and results without regenerating them. The scripts resolve paths relative to this repository, so the entry point can also be called from another working directory.

## Contents

| Path | Purpose |
| --- | --- |
| `run.py` | Entry point for numerical reproduction, validation, and optional figures |
| `analysis/inputs/` | Compact logged outcomes, task orders, distances, feature summaries, and provenance |
| `analysis/input_manifest.json` | SHA-256 hashes of the released compact inputs |
| `analysis/results/` | Per-order summaries, correlations, weighted scores, diagnostics, extended sum-of-squares decomposition, position-of-A associations, replicate-bootstrap leader probabilities, and generated tables |
| `analysis/scripts/recompute.py` | Rebuild all reported numerical summaries and tables |
| `analysis/scripts/validate.py` | Independent metric, correlation, and variation checks |
| `analysis/scripts/figures.py` | Task-distance matrices, descriptor/performance scatter plots, and path comparisons |
| `analysis/scripts/task_distributions.py` | Measured task-feature distribution figure from compact summaries |
| `analysis/scripts/figure_style.py` | Shared Arial figure style |
| `analysis/scripts/extract_inputs.py` | Optional reconstruction from original evaluation logs and numeric archives |
| `analysis/scripts/rebuild_distances.py` | Optional reconstruction of candidate distances from numeric trajectories |
| `figures/` | Generated numerical PDF/SVG figures |

The manuscript text, editable PowerPoint authoring tools, model checkpoints, and large trajectory archives are outside this repository.

## Data conventions

- Task A is Urban Merge, B is Highway Merge, C is Intersection, and D is Roundabout. `orders.csv` defines all sequences; CSV order IDs start at zero, while displayed IDs start at one.
- `stage_task_results.csv` contains 4,608 evaluations from 1,152 stage logs. `stage` and `task` index presentation positions, from 1 to 4. Recover task identity using the sequence in `orders.csv`.
- The column named `seed` preserves the archive's `random_0`, `random_1`, and `random_2` directory labels. These are local training-replicate labels, not numerical RNG seeds or an assertion of matched initialisations across combinations.
- `performance_per_seed.csv` contains 1,152 metric values: four CT/BWT metrics for each of the 288 runs. Reported standard deviations use `ddof=1`.
- Feature counts describe pooled historical observations. Repeated time steps and potentially overlapping trajectory windows are dependent; these counts do not add independent training replicates.
- `distance_matrices.csv` retains the compatibility column `corrected` for the primary reconstructed distances. `legacy` and `symmetrized` columns support historical sensitivity comparisons.

## Analysis definitions

For a presentation-ordered stage–task error matrix `R`, current-task error is the mean diagonal. BWT averages the error change on previously learned tasks within each later stage, then averages those stage means. Final error averages the last row. FDE is in metres; MR is in percent; MR-BWT is in percentage points. Smaller BWT means a more favourable change from the initial reference and does not necessarily mean lower final error.

The primary transition descriptor is the least-squares slope of successive speed-Wasserstein distances. With four tasks, `D = (Step3 - Step1) / 2`: the middle transition has zero coefficient. Symmetric distances produce six tied descriptor levels in this benchmark. Spearman correlations retain these ties. Starting-task-adjusted associations correlate globally ranked variables after subtracting their starting-task group means.

Uniform scores average the evaluated order means. The supplementary policy normalises `D` over the evaluated order set and uses `1 - D_tilde` for CT and `D_tilde` for BWT, shared across algorithms. A constant descriptor uses uniform weights. Descriptor and weighting choices were exploratory; the correlations and weighted comparisons are descriptive.

The centred sum of squares has four components: order, algorithm, their interaction, and training variation within each order–algorithm combination. Replicate labels are not treated as a matched global factor. Within-combination relabelling leaves the decomposition unchanged. No significance tests or independent-order bootstrap intervals are produced.

Speed and acceleration Wasserstein distances use all retained observations. Speed KL/JS use common 100-bin histograms and the documented probability smoothing. MMD uses one fixed 2,000-observation sample per task, pooled standardisation, and an RBF kernel. Full parameters, selected sample indices, units, source hashes, and reconstruction checks are in `analysis/inputs/`.

## Optional reconstruction from the original archive

The compact inputs above are sufficient for normal use. Authors with the separate original archive can rebuild them:

```sh
python analysis/scripts/extract_inputs.py --archive "/path/to/NTU-CL Task Order"
python analysis/scripts/task_distributions.py --archive "/path/to/NTU-CL Task Order"
python analysis/scripts/recompute.py
python analysis/scripts/validate.py
```

The archive must retain `exp_results/`, `interaction_dataset/`, `outputs/`, and the historical `similarity_analysis/export_single/distance_matrices_normalized.npz`. Raw trajectory reconstruction reads numeric `trajectory.npy` members only. Feature caches are loaded with `allow_pickle=False` and checked against recorded hashes. Source files are read-only and remain outside this repository. Rebuilt inputs should be reviewed against `analysis/input_manifest.json`; an input mismatch is reported rather than silently accepted.

## Citation and licence

Please cite the accompanying manuscript and identify the commit used for code reproduction. The following entry uses the current manuscript's title and author list without implying acceptance or publication:

```bibtex
@unpublished{li_task_order,
  author = {Zirui Li and Yunlong Lin and Haohan Yang and Xiaocong Zhao and Chen Lv},
  title = {What is the Influence of Task Order in Continual Learning? An Empirical Study and Order-aware Evaluation for Motion Forecasting},
  note = {Manuscript},
  url = {https://lzrbit.github.io/task_order/}
}
```

The repository itself may be cited as:

```bibtex
@misc{task_order_analysis,
  title = {Task Order in Continual-Learning Evaluation: Analysis Code},
  howpublished = {\url{https://github.com/lzrbit/task_order}},
  note = {Evaluation outcomes and post-training analysis code}
}
```

A software licence has not yet been specified by the authors.

## October 2026 update

`recompute.py` now also writes `extended_decomposition.csv` (starting-task split of the order term, with and without A-first orders, and for the warm-start current-task error), `position_correlations.csv` and `position_of_A_means.csv` (associations and means by the presentation position of task A), `bootstrap_summary.csv` and `per_order_leaders.csv` (replicate-level bootstrap with 5,000 draws and seed 0; separable per-order leaders), and `table_decomposition.tex`. `validate.py` checks these outputs, `teaser.py` draws the replicate-aware Figure 1, and `inputs/exploratory_scan_correlations.csv` records the 300-candidate exploratory descriptor scan. The 17 frozen inputs and the main CT/BWT values are unchanged.
