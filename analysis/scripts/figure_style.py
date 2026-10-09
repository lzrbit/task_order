"""Arial-only vector figure defaults for the IEEE two-column manuscript."""
import matplotlib as mpl
from matplotlib import font_manager

SINGLE = 3.5
DOUBLE = 7.16
INK = '#222222'
MUTED = '#666666'
BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#009E73'
ALGORITHMS = ['vanilla', 'der', 'syrem', 'gsm']
ALGO_COLOR = {'vanilla': '#353535', 'der': BLUE, 'syrem': ORANGE, 'gsm': GREEN}
ALGO_LABEL = {'vanilla': 'Vanilla', 'der': 'DER', 'syrem': 'SyReM', 'gsm': 'GSM'}
ALGO_MARKER = {'vanilla': 'o', 'der': 's', 'syrem': '^', 'gsm': 'D'}
TASK_COLOR = {'A': '#B8C9DD', 'B': '#B9D9CB', 'C': '#F3D49C', 'D': '#D2C4DF'}
TASK_LABEL = {'A': 'Urban merge', 'B': 'Highway merge', 'C': 'Intersection', 'D': 'Roundabout'}


def apply():
    # Fail explicitly if Arial is unavailable instead of substituting another face.
    for style, weight in [('normal', 'normal'), ('normal', 'bold'), ('italic', 'normal')]:
        font_manager.findfont(font_manager.FontProperties(family='Arial', style=style,
                              weight=weight), fallback_to_default=False)
    mpl.rcParams.update({
        'font.family': 'Arial', 'font.sans-serif': ['Arial'],
        'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
        'axes.titleweight': 'normal', 'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
        'text.color': INK, 'axes.labelcolor': INK,
        'axes.edgecolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.linewidth': .6, 'lines.linewidth': .85,
        'xtick.major.width': .5, 'ytick.major.width': .5,
        'xtick.major.size': 2, 'ytick.major.size': 2,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 300, 'savefig.facecolor': 'white',
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:style=italic', 'mathtext.bf': 'Arial:weight=bold',
        'mathtext.bfit': 'Arial:style=italic:weight=bold',
        'mathtext.sf': 'Arial', 'mathtext.tt': 'Arial', 'mathtext.cal': 'Arial',
        'mathtext.fallback': None,
    })
