#!/usr/bin/env python3
"""Build task-distribution summaries from frozen, public compact inputs.

Normal use needs no original trajectory archive:
    python task_distributions.py --preview-dir /path/to/previews

To reconstruct the compact inputs from the previously verified feature caches:
    python task_distributions.py --archive "/path/to/NTU-CL Task Order"

Archive mode checks every cache against trajectory_source_sha256.csv before
loading numeric arrays with allow_pickle=False. These are pooled observations,
not independent training examples. Histograms are for visualisation only; the
paper's Wasserstein distances use the complete empirical scalar distributions.
The figure uses exact sample quantiles rather than approximating them from bins.
All tasks share one horizontal scale for each feature. Thin intervals show the
5th--95th percentiles, coloured bands the 25th--75th, and dots the median.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PAPER = BASE.parent
INPUTS = BASE / 'inputs'
FEATURES = ['abs_speed', 'acceleration']
HISTOGRAM = INPUTS / 'task_distribution_histograms.csv'
SUMMARY = INPUTS / 'task_distribution_summary.csv'
PROVENANCE = INPUTS / 'task_distribution_provenance.json'
COLORS = {'A': '#4C78A8', 'B': '#32976E', 'C': '#D89A20', 'D': '#9373AC'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild_inputs(archive):
    params_path = INPUTS / 'distance_parameters.json'
    ledger_path = INPUTS / 'trajectory_source_sha256.csv'
    verification_path = INPUTS / 'raw_feature_verification.csv'
    params = json.loads(params_path.read_text())
    ledger = pd.read_csv(ledger_path).set_index('cached_features')
    verified = pd.read_csv(verification_path).set_index(['dataset', 'channel'])
    expected_counts = pd.read_csv(INPUTS / 'feature_counts.csv').set_index('scenario')
    distributions, source_checks = {}, []
    for scenario in params['canonical_scenarios']:
        parts = {feature: [] for feature in FEATURES}
        for dataset in params['groups'][scenario]:
            relative = f'outputs/train_{dataset}_combined_features.npz'
            path = archive / relative
            actual_hash = sha256(path)
            expected_hash = ledger.loc[relative, 'cache_sha256']
            if actual_hash != expected_hash:
                raise ValueError(f'Feature-cache SHA256 mismatch: {path}')
            check = {'scenario': scenario, 'dataset': dataset,
                     'cache_path': relative, 'cache_sha256': actual_hash,
                     'expected_cache_sha256': expected_hash, 'hash_matches': True}
            with np.load(path, allow_pickle=False) as cache:
                for feature in FEATURES:
                    values = cache[feature]
                    if values.ndim != 1 or values.dtype.hasobject or not np.isfinite(values).all():
                        raise ValueError(f'Invalid numeric feature array: {path} / {feature}')
                    audit = verified.loc[(dataset, feature)]
                    assert bool(audit.bitwise_equal)
                    assert len(values) == int(audit.n_features)
                    assert hashlib.sha256(values.tobytes()).hexdigest() == audit.derived_array_sha256
                    parts[feature].append(values)
                    check[feature + '_observations'] = len(values)
            source_checks.append(check)
        distributions[scenario] = {feature: np.concatenate(arrays)
                                   for feature, arrays in parts.items()}
        for feature in FEATURES:
            assert len(distributions[scenario][feature]) == int(
                expected_counts.loc[scenario, feature + '_observations'])

    pooled_min = min(x['abs_speed'].min() for x in distributions.values())
    pooled_max = max(x['abs_speed'].max() for x in distributions.values())
    assert pooled_min >= 0
    bins = int(params['histogram_bins'])
    lo, hi = params['acceleration_filter_inclusive']
    edges = {'abs_speed': np.linspace(pooled_min, pooled_max, bins + 1),
             'acceleration': np.linspace(lo, hi, bins + 1)}
    reconstruction = json.loads((INPUTS / 'distance_reconstruction.json').read_text())
    assert np.array_equal(edges['abs_speed'], reconstruction['histogram_edges'])
    rows, summaries = [], []
    for task, scenario in zip('ABCD', params['canonical_scenarios']):
        for feature in FEATURES:
            values = distributions[scenario][feature]
            assert values.min() >= edges[feature][0] and values.max() <= edges[feature][-1]
            counts, boundaries = np.histogram(values, bins=edges[feature])
            assert counts.sum() == len(values)
            probability = counts / len(values)
            widths = np.diff(boundaries)
            density = probability / widths
            assert np.isclose(np.dot(density, widths), 1, atol=1e-14)
            for index, count in enumerate(counts):
                rows.append({'task': task, 'scenario': scenario, 'feature': feature,
                             'bin': index, 'bin_left': boundaries[index],
                             'bin_right': boundaries[index+1], 'count': int(count),
                             'probability': probability[index], 'density': density[index],
                             'cumulative_probability': probability[:index+1].sum()})
            quantiles = np.quantile(values, [.05, .25, .5, .75, .95])
            summaries.append({'task': task, 'scenario': scenario, 'feature': feature,
                              'units': 'm/s' if feature == 'abs_speed' else 'm/s^2',
                              'n_observations': len(values), 'minimum': values.min(),
                              'maximum': values.max(), 'mean': values.mean(),
                              'sd_population': values.std(ddof=0),
                              **dict(zip(['q05', 'q25', 'median', 'q75', 'q95'], quantiles)),
                              'histogram_count_sum': int(counts.sum()),
                              'histogram_density_integral': float(np.dot(density, widths))})
    pd.DataFrame(rows).to_csv(HISTOGRAM, index=False)
    pd.DataFrame(summaries).to_csv(SUMMARY, index=False)
    metadata = {
        'purpose': 'Visualise task feature distributions; not an additional distance estimator.',
        'source': 'Previously bitwise-verified numeric trajectory feature caches.',
        'units': {'abs_speed': 'm/s', 'acceleration': 'm/s^2'},
        'task_groups': params['groups'],
        'speed_definition': 'Magnitude of target-agent velocity at all nine history steps.',
        'acceleration_definition': 'Adjacent speed differences divided by 0.1 seconds, retained inclusively within [-5, 5] m/s^2.',
        'observation_weighting': 'Every retained scalar observation has equal weight within its task; each task density integrates to one.',
        'dependence': 'Repeated time steps and potentially overlapping trajectory windows are not independent observations. No confidence intervals or significance tests are inferred from observation counts.',
        'binning': {'bins_per_feature': bins,
                    'speed_range': [float(pooled_min), float(pooled_max)],
                    'speed_edges': 'Identical to the shared speed histogram edges in distance_reconstruction.json.',
                    'acceleration_range': [lo, hi],
                    'boundary_convention': 'Left-closed/right-open, with final bin including its right endpoint.',
                    'smoothing': 'None; raw counts divided by task observation count and bin width.'},
        'summary_quantiles': 'Exact sample quantiles computed by numpy.quantile, default linear interpolation.',
        'inputs': {'distance_parameters_sha256': sha256(params_path),
                   'trajectory_source_ledger_sha256': sha256(ledger_path),
                   'raw_feature_verification_sha256': sha256(verification_path)},
        'compact_outputs': {HISTOGRAM.name: sha256(HISTOGRAM), SUMMARY.name: sha256(SUMMARY)},
        'all_12_cache_hashes_match': len(source_checks) == 12 and all(x['hash_matches'] for x in source_checks),
        'all_24_feature_hashes_match_prior_raw_trajectory_reconstruction': True,
        'source_checks': source_checks,
    }
    PROVENANCE.write_text(json.dumps(metadata, indent=2) + '\n')


def validate_compact_inputs(hist, summary):
    assert len(hist) == 800 and len(summary) == 8
    expected_groups = {(task, feature) for task in 'ABCD' for feature in FEATURES}
    assert set(zip(summary.task, summary.feature)) == expected_groups
    assert set(zip(hist.task, hist.feature)) == expected_groups
    for (task, feature), group in hist.groupby(['task', 'feature']):
        group = group.sort_values('bin')
        row = summary[(summary.task == task) & (summary.feature == feature)].iloc[0]
        counts = group['count'].to_numpy()
        widths = group.bin_right.to_numpy() - group.bin_left.to_numpy()
        assert (widths > 0).all() and (counts >= 0).all()
        assert counts.sum() == int(row.n_observations)
        assert np.allclose(group.probability, counts / counts.sum(), atol=1e-15)
        assert np.allclose(group.density * widths, group.probability, atol=1e-15)
        assert np.isclose(np.dot(group.density, widths), 1, atol=1e-14)
        assert np.allclose(np.cumsum(group.probability), group.cumulative_probability, atol=1e-14)
        assert np.allclose(group.bin_right.iloc[:-1], group.bin_left.iloc[1:], atol=1e-15)
        quantiles = row[['minimum', 'q05', 'q25', 'median', 'q75', 'q95', 'maximum']].to_numpy(dtype=float)
        assert np.isfinite(quantiles).all() and (np.diff(quantiles) >= 0).all()


def draw(summary, preview_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.font_manager import FontProperties
    import figure_style as style

    style.apply()
    fig = plt.figure(figsize=(style.SINGLE, 2.6))
    handles = [Line2D([], [], color=style.INK, lw=.8, label='5–95%'),
               Patch(facecolor='#CBD6E0', edgecolor='#74879A', lw=.7, label='25–75%'),
               Line2D([], [], linestyle='none', marker='o', markersize=3.8,
                      markerfacecolor='white', markeredgecolor=style.INK,
                      markeredgewidth=.9, label='Median')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.515, .994),
               ncol=3, fontsize=6.8, handlelength=1.7, columnspacing=1.55,
               handletextpad=.55, borderpad=0)
    for index, feature in enumerate(FEATURES):
        bottom = .595 if index == 0 else .145
        ax = fig.add_axes([.307, bottom, .661, .300])
        rows = summary[summary.feature == feature].set_index('task').loc[list('ABCD')]
        # Limits encompass the displayed central 90% interval in every task.
        # Rounding only supplies readable ticks; no task-specific rescaling is used.
        if feature == 'abs_speed':
            upper = 5 * np.ceil(rows.q95.max() / 5)
            ax.set_xlim(0, upper)
            ax.set_xticks(np.arange(0, upper + .1, 5))
        else:
            upper = .5 * np.ceil(max(abs(rows.q05.min()), abs(rows.q95.max())) / .5)
            ax.set_xlim(-upper, upper)
            ax.set_xticks(np.linspace(-upper, upper, 5))
        ax.set_ylim(-.6, 3.6)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E2E6E9', linewidth=.5)
        if feature == 'acceleration':
            ax.axvline(0, color='#ADB6BD', lw=.7, zorder=0)
        for y, task in zip([3, 2, 1, 0], 'ABCD'):
            row = rows.loc[task]
            color = COLORS[task]
            ax.plot([row.q05, row.q95], [y, y], color=color, lw=.85,
                    solid_capstyle='butt', zorder=2)
            ax.vlines([row.q05, row.q95], y-.09, y+.09, color=color, lw=.75, zorder=2)
            ax.add_patch(Rectangle((row.q25, y-.15), row.q75-row.q25, .30,
                                   facecolor=color, edgecolor='none', alpha=.25, zorder=3))
            ax.add_patch(Rectangle((row.q25, y-.15), row.q75-row.q25, .30,
                                   facecolor='none', edgecolor=color, lw=.75, zorder=4))
            ax.plot(row['median'], y, 'o', markersize=4.0, markerfacecolor='white',
                    markeredgecolor=color, markeredgewidth=1.1, zorder=5)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([3, 2, 1, 0], [f'{task}  {style.TASK_LABEL[task]}' for task in 'ABCD'])
        ax.tick_params(axis='y', labelsize=6.9, length=0, pad=5)
        ax.tick_params(axis='x', labelsize=6.7, pad=2)
        # Keep the title bold while the parenthesised panel identifier is regular.
        prefix = f'({chr(97+index)})  '
        title = 'Target-agent speed (m/s)' if index == 0 else 'Acceleration (m/s²)'
        renderer = fig.canvas.get_renderer()
        regular = FontProperties(family='Arial', size=7.4, weight='normal')
        bold = FontProperties(family='Arial', size=7.4, weight='bold')
        prefix_width = renderer.get_text_width_height_descent(prefix, regular, False)[0]
        title_width = renderer.get_text_width_height_descent(title, bold, False)[0]
        left = .51 - (prefix_width + title_width) / (2 * fig.bbox.width)
        fig.text(left, bottom-.105, prefix.rstrip(), fontsize=7.4,
                 weight='normal', ha='left')
        fig.text(left + prefix_width / fig.bbox.width, bottom-.105, title,
                 fontsize=7.4, weight='bold', ha='left')
    out = PAPER / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ['pdf', 'svg']:
        path = out / f'fig_task_distributions.{suffix}'
        fig.savefig(path)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / 'fig_task_distributions.png', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Rebuild compact inputs from a local archive after verifying hashes')
    parser.add_argument('--preview-dir', type=Path, help='Write a PNG preview outside the repository')
    args = parser.parse_args()
    if args.archive:
        rebuild_inputs(args.archive)
    provenance = json.loads(PROVENANCE.read_text())
    for path in (HISTOGRAM, SUMMARY):
        assert sha256(path) == provenance['compact_outputs'][path.name]
    # Preserve exact floating endpoints, including the observed maximum speed.
    hist = pd.read_csv(HISTOGRAM, float_precision='round_trip')
    summary = pd.read_csv(SUMMARY, float_precision='round_trip')
    validate_compact_inputs(hist, summary)
    draw(summary, args.preview_dir)
    print('Rebuilt task quantile summaries; all eight input histograms and exact quantile orderings verified.')


if __name__ == '__main__':
    main()
