#!/usr/bin/env python3
"""Single source of truth for every paper figure: true-LaTeX (Computer Modern) fonts,
one CVD-validated categorical palette with fixed meaning, restrained marks and axes.
Every figure script does `from money_style import *; setup()` and uses these colors/helpers.

Palette meaning is FIXED across the paper (never repurposed):
  BLUE  = our method / after / full-event / reweighted
  RED   = before / none / no reweighting
  OCHRE = intermediate / hemisphere / parameters-only
  GREY  = statistical-noise floor / reference / neutral
Validated (scripts/validate_palette.js): all in the lightness band, chroma floor, CVD>=12,
contrast>=3:1 on a near-white surface.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE   = '#3366CC'
RED    = '#CC3355'
OCHRE  = '#997700'
GREY   = '#8A8D91'
TEAL   = '#009988'   # fourth series, added for the d=8 stage
INK    = '#1A1A1A'   # primary text / axes
MUTED  = '#5A5D62'   # secondary text / annotations
FAINT  = '#B9BCC0'   # gridlines if ever used

_PREAMBLE = r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}'


def setup(usetex=True):
    plt.rcParams.update({
        'text.usetex': usetex,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'text.latex.preamble': _PREAMBLE,
        'mathtext.fontset': 'cm',
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.labelsize': 17.5,
        'axes.linewidth': 0.7,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'text.color': INK,
        'axes.titlecolor': INK,
        'xtick.labelsize': 14.5, 'ytick.labelsize': 14.5,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'xtick.minor.size': 2.2, 'ytick.minor.size': 2.2,
        'legend.fontsize': 14, 'legend.frameon': False,
        'legend.handlelength': 1.3, 'legend.handletextpad': 0.6,
        'axes.grid': False,
        'figure.dpi': 130, 'savefig.dpi': 600,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'lines.linewidth': 2.0, 'lines.markersize': 7,
        'errorbar.capsize': 2.5,
    })


def trim(ax, top=True, right=True):
    """Remove top/right spines for a lighter frame (keep ticks in)."""
    if top:
        ax.spines['top'].set_visible(False); ax.tick_params(which='both', top=False)
    if right:
        ax.spines['right'].set_visible(False); ax.tick_params(which='both', right=False)


def ideal(ax, y=1.0, label=None):
    """The dashed unit line: 1 = matches a fresh run within noise."""
    ax.axhline(y, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1, label=label)


def floor_ticks(ax, x_positions, floor_vals, halfwidth=0.34, label='statistical floor'):
    """Solid black ticks marking each observable's own noise floor."""
    for x, f in zip(x_positions, floor_vals):
        ax.hlines(f, x - halfwidth, x + halfwidth, color=INK, lw=1.5, zorder=4)
    ax.hlines([], [], [], color=INK, lw=1.5, label=label)   # legend proxy


def panel_tag(ax, s, loc=(0.03, 0.92)):
    ax.text(loc[0], loc[1], s, transform=ax.transAxes, fontsize=11, va='top')


def panel_title(ax, s, fs=15.5):
    """One clear title above each panel, never inside the axes."""
    ax.set_title(s, loc='left', fontsize=fs, pad=7, color=INK)


def save(fig, path):
    fig.savefig(path)
    print(f'wrote {path}')
    plt.close(fig)


if __name__ == '__main__':
    setup()
    import numpy as np
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.linspace(0, 1, 20)
    ax.plot(x, x, '-', color=BLUE, label=r'our method')
    ax.plot(x, x**1.5, 's', color=RED, ms=4, label=r'baseline')
    ideal(ax, 0.5)
    ax.set_xlabel(r'held-out $\alpha_s$'); ax.set_ylabel(r'match width $\chi^2/\mathrm{ndf}$')
    ax.legend(); trim(ax)
    save(fig, 'output/_style_selftest.pdf')
