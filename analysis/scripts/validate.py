#!/usr/bin/env python3
"""Validate frozen inputs and independently reconstruct the main reported metrics."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
INPUTS, RESULTS = BASE / 'inputs', BASE / 'results'


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def check_inputs():
    manifest = json.loads((BASE / 'input_manifest.json').read_text())
    for relative, expected in manifest.items():
        path = BASE / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f'Frozen input changed: {relative}')
    return {'input_files_verified': len(manifest)}


def check_results():
    raw = pd.read_csv(INPUTS / 'stage_task_results.csv')
    performance = pd.read_csv(INPUTS / 'performance_per_seed.csv')
    orders = pd.read_csv(INPUTS / 'orders.csv').sort_values('Order')
    expected = performance.set_index(['seed', 'Order', 'algorithm', 'metric']).value
    require(len(raw) == 4608 and len(performance) == 1152, 'Unexpected evaluation coverage')
    require(len(orders) == 24 and orders.Order.nunique() == 24, 'Unexpected task-order coverage')
    require(not raw.duplicated(['seed', 'Order', 'algorithm', 'stage', 'task']).any(),
            'Duplicate stage/task observations')
    errors = []
    post = np.zeros((4, 4))
    post[1, 0], post[2, :2], post[3, :3] = 1/3, 1/6, 1/9
    reference = np.array([11/18, 5/18, 1/9, 0])
    for key, group in raw.groupby(['seed', 'Order', 'algorithm']):
        for metric in ['FDE', 'MR']:
            matrix = group.pivot(index='stage', columns='task', values=metric).sort_index().sort_index(axis=1).to_numpy()
            require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), 'Incomplete performance matrix')
            # Closed-form coefficients provide a separate check of stagewise aggregation.
            derived = {'CT': np.trace(matrix)/4,
                       'BWT': np.sum(post*matrix)-np.dot(reference, np.diag(matrix))}
            for kind, value in derived.items():
                errors.append(abs(value-expected.loc[key+(metric+'-'+kind,)]))
    require(max(errors) < 1e-10, 'CT/BWT summaries disagree with stage/task records')
    per_order = pd.read_csv(RESULTS / 'per_order.csv').set_index('Order')
    metric_means = performance.groupby(['Order', 'metric', 'algorithm']).value.mean()
    for (order, metric, algorithm), value in metric_means.items():
        require(np.isclose(value, per_order.loc[order, metric+'__'+algorithm], rtol=1e-12, atol=1e-12),
                'Order means disagree with training replicates')
    require(np.allclose(per_order.D, (per_order.Step3-per_order.Step1)/2), 'Descriptor formula mismatch')
    require(per_order.D.round(10).nunique() == 6, 'Unexpected descriptor ties')
    correlations = pd.read_csv(RESULTS / 'correlations.csv')
    correlation_errors = [abs(spearmanr(per_order.D, per_order[f'{r.metric}__{r.algorithm}']).statistic-r.rho_all)
                          for r in correlations.itertuples()]
    require(len(correlation_errors) == 16 and max(correlation_errors) < 1e-12,
            'Descriptor correlations disagree with order means')
    ss = pd.read_csv(RESULTS / 'sum_of_squares.csv')
    require(set(ss.component) == {'order', 'algorithm', 'order_x_algorithm', 'within_cell_training'},
            'Unexpected sum-of-squares decomposition')
    for metric, group in performance.groupby('metric'):
        cell_mean = group.groupby(['Order', 'algorithm']).value.transform('mean')
        within = ((group.value-cell_mean)**2).sum()
        total = ((group.value-group.value.mean())**2).sum()
        measured = ss[ss.metric == metric].set_index('component')
        require(np.isclose(within, measured.loc['within_cell_training', 'SS'], rtol=1e-12),
                'Within-cell variation mismatch')
        require(np.isclose(measured.SS.sum(), total, rtol=1e-12), 'SS components do not sum to total')
    # Extended decomposition: the starting-task and within-start parts must recombine to the order term,
    # and every row of fractions must sum to one.
    extended = pd.read_csv(RESULTS / 'extended_decomposition.csv')
    require(np.allclose(extended.start + extended.order_within_start, extended.order, atol=1e-12),
            'Starting-task split does not recombine to the order term')
    require(np.allclose(extended.order + extended.algorithm + extended.order_x_algorithm
                        + extended.within_cell_training, 1, atol=1e-12),
            'Extended decomposition fractions do not sum to one')
    for metric in ['FDE-CT', 'MR-CT', 'FDE-BWT', 'MR-BWT']:
        full = ss[ss.metric == metric].set_index('component')
        row = extended[(extended.metric == metric) & (extended.subset == 'all_orders')].iloc[0]
        require(np.isclose(row.order, full.loc['order', 'fraction_total_SS'], atol=1e-12),
                'Extended decomposition disagrees with sum_of_squares.csv')
    # Position-of-A associations: the marginal descriptor correlation must match correlations.csv,
    # and the sign of D must coincide with task A occupying positions 1-2.
    position = pd.read_csv(RESULTS / 'position_correlations.csv')
    sequences = orders.set_index('Order').scenario_sequence
    pos_a = np.array([s.split('|').index('urban_merge') + 1 for s in sequences.reindex(range(24))])
    require(all((per_order.D.reindex(range(24)).values < 0) == (pos_a <= 2)),
            'sign(D) no longer coincides with task A in positions 1-2')
    for r in correlations.itertuples():
        match = position[(position.metric == r.metric) & (position.algorithm == r.algorithm)]
        require(len(match) == 1 and abs(float(match.rho_D.iloc[0]) - r.rho_all) < 1e-12,
                'Position-adjusted table disagrees with correlations.csv')
    # Bootstrap summary: leader probabilities form a distribution over algorithms for every metric.
    boot = pd.read_csv(RESULTS / 'bootstrap_summary.csv')
    for metric, group in boot.groupby('metric'):
        require(np.isclose(group.p_best_uniform.sum(), 1, atol=1e-9), f'Uniform leader probabilities do not sum to one: {metric}')
        require((group.winner_count_q025 <= group.winner_count).all() and (group.winner_count <= group.winner_count_q975).all(),
                f'Winner count outside its bootstrap interval: {metric}')
    return {'stage_task_rows': len(raw), 'training_runs': len(raw.groupby(['seed', 'Order', 'algorithm'])),
            'main_metric_rows_checked': len(errors), 'max_CT_BWT_difference': max(errors),
            'correlations_checked': len(correlation_errors), 'max_correlation_difference': max(correlation_errors),
            'distinct_descriptor_levels': 6, 'sum_of_squares_components': list(ss.component.unique()),
            'extended_decomposition_rows': len(extended), 'position_correlation_rows': len(position),
            'bootstrap_metrics': int(boot.metric.nunique())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = check_inputs()
    if not args.inputs_only:
        report.update(check_results())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
