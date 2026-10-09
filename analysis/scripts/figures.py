"""Build manuscript Figures 5-7 from audited tabular inputs, using Arial only.

Run recompute.py first. This entry point writes only the distance comparison,
descriptor scatter plots, and extreme-path comparison (PDF and editable SVG).
Figures 1-2 have a separate schematic build and are never overwritten here.
Use --preview-dir PATH to write PNG previews outside the manuscript directory.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
import figure_style as st

BASE = Path(__file__).resolve().parents[1]
PAPER = BASE.parent
RESULTS = BASE / 'results'
INPUTS = BASE / 'inputs'
METRICS = ['FDE-CT', 'FDE-BWT', 'MR-CT', 'MR-BWT']
SEQ_CMAP = LinearSegmentedColormap.from_list('paper_blue', ['#F6F8FA', '#B8D7E5', '#5594B8', '#12496B'])
# Dark task hues correspond to the pale task tiles, independently of algorithm colours.
TASK_TEXT_COLOR = {'A': '#35648B', 'B': '#237550', 'C': '#9F6500', 'D': '#76558F'}


def seq(value):
    """Accept compact or separator-delimited sequences, including scenario names."""
    letters = re.findall(r'[ABCD]', str(value))
    if len(letters) == 4:
        return letters
    names = {'urban_merge': 'A', 'highway_merge': 'B', 'intersection': 'C', 'roundabout': 'D'}
    found = re.findall('urban_merge|highway_merge|intersection|roundabout', str(value))
    if len(found) == 4:
        return [names[x] for x in found]
    raise ValueError(f'Cannot parse sequence: {value}')


def save(fig, name, preview_dir):
    out = PAPER / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'svg']:
        path = out / f'{name}.{extension}'
        fig.savefig(path, bbox_inches=None)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f'{name}.png', dpi=240)
    plt.close(fig)


def text(ax, x, y, s, size=7, **kw):
    return ax.text(x, y, s, fontsize=size, ha=kw.pop('ha', 'left'),
                   va=kw.pop('va', 'center'), **kw)


def panel_label(fig, x, y, letter, title, size):
    """Centre a normal-weight parenthesised identifier beside a bold title."""
    from matplotlib.font_manager import FontProperties
    renderer = fig.canvas.get_renderer()
    regular = FontProperties(family='Arial', size=size, weight='normal')
    bold = FontProperties(family='Arial', size=size, weight='bold')
    prefix = f'({letter})  '
    prefix_width = renderer.get_text_width_height_descent(prefix, regular, False)[0]
    title_width = renderer.get_text_width_height_descent(title, bold, False)[0]
    left = x - (prefix_width + title_width) / (2 * fig.bbox.width)
    fig.text(left, y, prefix.rstrip(), fontsize=size, ha='left', weight='normal')
    fig.text(left + prefix_width / fig.bbox.width, y, title,
             fontsize=size, ha='left', weight='bold')


def tiles(ax, sequence, x, y, w=.11, h=.23, gap=.064):
    for i, task in enumerate(sequence):
        xx = x + i * (w + gap)
        ax.add_patch(Rectangle((xx, y-h/2), w, h, facecolor=st.TASK_COLOR[task],
                               edgecolor='white', lw=.45))
        text(ax, xx+w/2, y, task, 7, ha='center')
        if i < 3:
            ax.add_patch(FancyArrowPatch((xx+w+.006, y), (xx+w+gap-.006, y),
                         arrowstyle='-|>', mutation_scale=5, lw=.5,
                         color=st.MUTED, shrinkA=0, shrinkB=0))


def matrix_from_frame(frame, key):
    sub = frame[frame.distance == key].copy()
    mapping = {'urban_merge': 'A', 'highway_merge': 'B', 'intersection': 'C', 'roundabout': 'D'}
    sub['from'] = sub['from'].replace(mapping)
    sub['to'] = sub['to'].replace(mapping)
    return sub.pivot(index='from', columns='to', values='corrected').reindex(
        index=list('ABCD'), columns=list('ABCD')).to_numpy(float)


def distances(frame, preview):
    definitions = [('Wasserstein-1', 'Target speed', 'abs_speed__wasserstein_distance'),
                   ('KL divergence', 'Target speed; directed', 'abs_speed__kl_divergence'),
                   ('JS divergence', 'Target speed', 'abs_speed__jensen-shannon_divergence'),
                   ('MMD', 'Target speed', 'abs_speed__mmd'),
                   ('Wasserstein-1', 'Acceleration', 'acceleration__wasserstein_distance')]
    fig = plt.figure(figsize=(st.DOUBLE, 2.00))
    for j, (title, subtitle, key) in enumerate(definitions):
        left = .032 + j * .183
        ax = fig.add_axes([left, .355, .153, .548])
        m = matrix_from_frame(frame, key)
        mask = ~np.eye(4, dtype=bool)
        v = m[mask]
        z = (m-v.min())/(v.max()-v.min())
        np.fill_diagonal(z, np.nan)
        ax.pcolormesh(np.arange(5)-.5, np.arange(5)-.5, np.ma.masked_invalid(z),
                      cmap=SEQ_CMAP, vmin=0, vmax=1, linewidth=0, rasterized=False)
        ax.set_xlim(-.5, 3.5)
        ax.set_ylim(3.5, -.5)
        ax.set_aspect('equal')
        for r in range(4):
            for c in range(4):
                if r == c:
                    text(ax, c, r, '-', 6.7, ha='center', color='#ADB4B9')
                else:
                    text(ax, c, r, f'{z[r,c]:.2f}', 6.7, ha='center',
                         color='white' if z[r,c] > .66 else st.INK)
        ax.set_xticks(range(4), list('ABCD'))
        ax.set_yticks(range(4), list('ABCD'))
        ax.tick_params(length=0, pad=3, labelsize=6.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which='minor', color='white', lw=.6)
        ax.tick_params(which='minor', length=0)
        # Each panel identifier and title sit below its matrix, after the ticks.
        panel_label(fig, left+.0765, .220, chr(97+j), title, 7.2)
        fig.text(left+.0765, .150, subtitle, fontsize=6.7, ha='center', color=st.MUTED)
    cax = fig.add_axes([.950, .375, .012, .505])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=SEQ_CMAP),
                      cax=cax, ticks=[0, .5, 1])
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.ax.tick_params(labelsize=6.4, length=2, pad=2)
    fig.text(.5, .032, 'Within-panel min-max scaling; diagonal omitted',
             fontsize=6.5, ha='center', color=st.MUTED)
    save(fig, 'fig_distance_comparison', preview)


def extreme_paths(data, paths, preview):
    fig = plt.figure(figsize=(st.DOUBLE, 3.40))
    for j, task in enumerate('ABCD'):
        ax = fig.add_axes([.025+j*.247, .720, .218, .255])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        text(ax, .535, .94, f'Starting task {task}', 7.2, ha='center', weight='bold',
             color=TASK_TEXT_COLOR[task])
        for kind, y, color in [('max', .63, st.ORANGE), ('min', .25, st.BLUE)]:
            row = paths[(paths.first_task == task) & (paths.kind == kind)].iloc[0]
            text(ax, .025, y, kind.capitalize(), 6.8, color=color)
            tiles(ax, seq(row.Sequence), .260, y)
    # The complete strip of selected orders forms panel a.
    panel_label(fig, .5, .675, 'a', 'Selected task orders', 7.6)
    for i, metric in enumerate(METRICS):
        left = .073+i*.240
        ax = fig.add_axes([left, .265, .177, .315])
        per = data.set_index('Order')[[f'{metric}__{x}' for x in st.ALGORITHMS]].mean(axis=1)
        standardized = (per-per.mean())/per.std(ddof=0)
        maximum, minimum = [], []
        for j, task in enumerate('ABCD'):
            pair = {kind: paths[(paths.first_task == task) & (paths.kind == kind)].iloc[0]
                    for kind in ['max', 'min']}
            lo = float(standardized.loc[pair['min'].Order])
            hi = float(standardized.loc[pair['max'].Order])
            minimum.append(lo)
            maximum.append(hi)
            ax.plot([lo, hi], [j, j], color='#AAB2B8', lw=.8, zorder=1)
        ax.scatter(maximum, range(4), color=st.ORANGE, s=22, marker='o', alpha=.55, zorder=3)
        ax.scatter(minimum, range(4), color=st.BLUE, s=22, marker='s', alpha=.55, zorder=3)
        ax.axvline(0, color='#D5DADF', lw=.5, zorder=0)
        ax.set_yticks(range(4), list('ABCD'))
        ax.set_ylim(3.55, -.55)
        ax.set_xlabel('Standardised metric', fontsize=6.8, labelpad=3)
        if i == 0:
            ax.set_ylabel('Starting task', fontsize=6.8, labelpad=3)
        ax.tick_params(labelsize=6.6)
        ax.margins(x=.16)
        panel_label(fig, left+.0885, .105, chr(98+i), metric, 7.5)
    handles = [Line2D([], [], ls='none', marker='o', color=st.ORANGE, ms=4, alpha=.55,
                      label='Maximum cumulative shift'),
               Line2D([], [], ls='none', marker='s', color=st.BLUE, ms=4, alpha=.55,
                      label='Minimum cumulative shift')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(.52, .008), fontsize=6.9, columnspacing=2)
    save(fig, 'fig_max_min_paths', preview)


def correlation(data, corr, preview):
    """Show per-algorithm Spearman associations while preserving tied D values."""
    metrics = ['FDE-CT', 'MR-CT', 'FDE-BWT', 'MR-BWT']
    units = {'FDE-CT': 'FDE-CT (m)', 'MR-CT': 'MR-CT (%)',
             'FDE-BWT': 'FDE-BWT (m)', 'MR-BWT': 'MR-BWT (pp)'}
    fig = plt.figure(figsize=(st.DOUBLE, 5.15))
    levels = np.sort(data.D.unique())
    for col, algorithm in enumerate(st.ALGORITHMS):
        left = .074 + col*.239
        fig.text(left+.088, .982, st.ALGO_LABEL[algorithm], fontsize=8,
                 ha='center', va='top', weight='bold', color=st.ALGO_COLOR[algorithm])
        for row, metric in enumerate(metrics):
            bottom = .787-row*.220
            ax = fig.add_axes([left, bottom, .176, .154])
            for d in levels:
                ax.axvline(d, color='#E6EAED', lw=.4, zorder=0)
            y = data[f'{metric}__{algorithm}'].to_numpy()
            ax.scatter(data.D, y, c=st.ALGO_COLOR[algorithm],
                       marker=st.ALGO_MARKER[algorithm], s=12, alpha=.68,
                       edgecolor='white', lw=.25, zorder=3)
            rho = corr[(corr.metric == metric) & (corr.algorithm == algorithm)].rho_all
            assert len(rho) == 1
            ax.text(.96, .93 if metric.endswith('CT') else .08,
                    f'ρ = {float(rho.iloc[0]):+.2f}'.replace('-', '−'),
                    transform=ax.transAxes, fontsize=6.3, ha='right',
                    va='top' if metric.endswith('CT') else 'bottom',
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .7, 'alpha': .9})
            all_y = data[[f'{metric}__{a}' for a in st.ALGORITHMS]].to_numpy()
            span = np.ptp(all_y)
            ax.set_ylim(all_y.min()-.08*span, all_y.max()+.12*span)
            ax.set_xlim(-1.08, 1.08)
            ax.set_xticks([-1, 0, 1])
            ax.tick_params(labelsize=6.1, pad=2)
            if col == 0:
                ax.set_ylabel(units[metric], fontsize=7, labelpad=3)
            fig.text(left+.088, bottom-.049, f'({chr(97+row*4+col)})',
                     fontsize=7.5, ha='center', weight='normal')
    fig.text(.52, .022, 'Transition-direction descriptor, D(o) (m/s)', fontsize=7.2, ha='center')
    save(fig, 'fig_unified_D_main', preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path, help='Optional PNG preview directory')
    args = parser.parse_args()
    st.apply()
    data = pd.read_csv(RESULTS/'per_order.csv')
    corr = pd.read_csv(RESULTS/'correlations.csv')
    dist = pd.read_csv(INPUTS/'distance_matrices.csv')
    paths = pd.read_csv(RESULTS/'extreme_paths.csv')
    paths['first_task'] = paths['first_task'].replace(
        {'urban_merge': 'A', 'highway_merge': 'B', 'intersection': 'C', 'roundabout': 'D'})
    corr = corr[corr.distance == 'abs_speed__wasserstein_distance']
    assert len(data) == 24 and data.Order.nunique() == 24
    assert np.allclose(data.D, (data.Step3-data.Step1)/2)
    assert data.D.round(10).nunique() <= 6
    distances(dist, args.preview_dir)
    correlation(data, corr, args.preview_dir)
    extreme_paths(data, paths, args.preview_dir)
    print('Rebuilt Figures 5-7 as Arial-only PDF and SVG; Figures 1-2 use their separate schematic build.')


if __name__ == '__main__':
    main()
