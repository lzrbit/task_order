"""Build Figure 1 (fig_teaser_v2): a replicate-aware ranking comparison for two orders.

Panel (a): mean FDE-CT and ranks for two orders that differ only in their final two
tasks; bars span one sample standard deviation on either side, and in both orders the
leader's gap to the runner-up exceeds the pooled replicate standard deviation.
Panel (b): the same runs split into the first-stage error R_{1,1} and the warm-start
error CT_{2:N}, showing that the CT reversal arises in the first stage.
Run recompute.py first. The author-supplied fig_teaser.pdf is never overwritten.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figure_style as st
from figures import save, seq, tiles, text, panel_label

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / 'results'
INPUTS = BASE / 'inputs'
ORDERS = ['ACBD', 'ACDB']
METRIC = 'FDE-CT'


def stage_stats(raw, order_id):
    """Per-algorithm mean and SD of CT, R_{1,1} and CT_{2:N} (FDE) for one order."""
    out = {}
    for a in st.ALGORITHMS:
        rows = []
        for seed, g in raw[(raw.Order == order_id) & (raw.algorithm == a)].groupby('seed'):
            M = g.pivot(index='stage', columns='task', values='FDE').values
            d = np.diag(M)
            rows.append((d.mean(), d[0], d[1:].mean()))
        rows = np.array(rows)
        out[a] = {'ct': (rows[:, 0].mean(), rows[:, 0].std(ddof=1)),
                  'r11': (rows[:, 1].mean(), rows[:, 1].std(ddof=1)),
                  'ct2n': (rows[:, 2].mean(), rows[:, 2].std(ddof=1))}
    return out


def slope_panel(ax, stats, key, xs, label_fmt='{:.2f}', ranks=True, size=6.4):
    for a in st.ALGORITHMS:
        m = [stats[o][a][key][0] for o in ORDERS]
        s = [stats[o][a][key][1] for o in ORDERS]
        ax.plot(xs, m, color=st.ALGO_COLOR[a], lw=.9, zorder=2, alpha=.9)
        ax.errorbar(xs, m, yerr=s, fmt='none', ecolor=st.ALGO_COLOR[a], elinewidth=.7,
                    capsize=1.6, capthick=.5, alpha=.75, zorder=2)
        ax.scatter(xs, m, color=st.ALGO_COLOR[a], marker=st.ALGO_MARKER[a], s=15,
                   edgecolor='white', lw=.4, zorder=4)
    if ranks:
        span = ax.get_ylim()
        gap = .06 * abs(span[1] - span[0])
        for j, o in enumerate(ORDERS):
            order = pd.Series({a: stats[o][a][key][0] for a in st.ALGORITHMS}).sort_values()
            pos = order.values.copy()
            for k in range(1, len(pos)):
                pos[k] = max(pos[k], pos[k-1] + gap)
            for rank, (a, value) in enumerate(order.items(), start=1):
                lab = f'{rank}  {st.ALGO_LABEL[a]}  ' + label_fmt.format(value)
                if j == 0:
                    text(ax, xs[0]-.08, pos[rank-1], lab, size, ha='right', color=st.ALGO_COLOR[a])
                else:
                    text(ax, xs[1]+.08, pos[rank-1], lab, size, ha='left', color=st.ALGO_COLOR[a])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path, help='Optional PNG preview directory')
    args = parser.parse_args()
    st.apply()
    raw = pd.read_csv(INPUTS / 'stage_task_results.csv')
    orders = pd.read_csv(INPUTS / 'orders.csv').sort_values('Order')
    letters = {int(r.Order): ''.join(seq(r.scenario_sequence)) for r in orders.itertuples()}
    ids = {v: k for k, v in letters.items()}
    leaders = pd.read_csv(RESULTS / 'per_order_leaders.csv')
    stats = {o: stage_stats(raw, ids[o]) for o in ORDERS}
    for o in ORDERS:
        row = leaders[(leaders.metric == METRIC) & (leaders.Sequence == o)].iloc[0]
        assert bool(row.separable), f'{o}: leader not separable from runner-up'
    fig = plt.figure(figsize=(st.SINGLE, 3.15))
    # Panel (a): FDE-CT slopegraph with ranks.
    ax = fig.add_axes([.14, .47, .72, .39])
    xs = [0, 1]
    ax.set_xlim(-.9, 1.9)
    ax.set_ylim(11.7, 2.3)
    slope_panel(ax, stats, 'ct', xs)
    ax.set_xticks([])
    ax.set_ylabel('Mean FDE-CT (m)', fontsize=7, labelpad=2)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', labelsize=6.3)
    for j, o in enumerate(ORDERS):
        strip = fig.add_axes([.02 if j == 0 else .66, .88, .32, .11])
        strip.set_xlim(0, 1)
        strip.set_ylim(0, 1)
        strip.axis('off')
        tiles(strip, list(o), .12, .40, w=.13, h=.52, gap=.07)
        text(strip, .50, .99, f'Order {o[0]}–{o[1]}–{o[2]}–{o[3]}', 6.8,
             ha='center', va='top', color=st.MUTED)
    panel_label(fig, .5, .415, 'a', 'Current-task error and ranking', 7.2)
    # Panel (b): the first-stage and warm-start parts of the same runs.
    for k, (key, title, lim) in enumerate([('r11', 'First stage, $R_{1,1}$ (m)', (46, 0)),
                                           ('ct2n', 'Warm start, $\\mathrm{CT}_{2:N}$ (m)', (2.3, 0.5))]):
        axb = fig.add_axes([.14 + k*.44, .13, .30, .20])
        axb.set_xlim(-.35, 1.35)
        axb.set_ylim(*lim)
        slope_panel(axb, stats, key, xs, ranks=False)
        axb.set_xticks(xs, [o[0]+'–'+o[1]+'–'+o[2]+'–'+o[3] for o in ORDERS], fontsize=6)
        axb.tick_params(axis='y', labelsize=6.1)
        axb.set_title(title, fontsize=6.8, pad=3)
    panel_label(fig, .5, .022, 'b', 'First-stage and warm-start components', 7.2)
    save(fig, 'fig_teaser_v2', args.preview_dir)
    print('Rebuilt fig_teaser_v2 (PDF and SVG); the author-supplied fig_teaser.pdf is unchanged.')


if __name__ == '__main__':
    main()
