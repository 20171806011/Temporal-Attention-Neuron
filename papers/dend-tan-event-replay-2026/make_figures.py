"""Build manuscript figures from the frozen schedule and published block effects."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
NAVY, TEAL, GREY, RED = '#24334b', '#176c67', '#727a88', '#ad4538'
METADATA = {'CreationDate': datetime(2026, 11, 11, tzinfo=timezone.utc), 'Creator': 'Manuscript figure source'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                     'text.color': NAVY, 'axes.labelcolor': NAVY, 'xtick.color': NAVY, 'ytick.color': NAVY})


def schedule(path):
    fig, ax = plt.subplots(figsize=(7.2, 2.12), layout='constrained')
    ax.set(xlim=(-.8, 36), ylim=(-.37, 1.65), yticks=[0, 1],
           yticklabels=['Replay input', 'Online input'], xlabel='Time step')
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
    ax.tick_params(axis='y', length=0)
    for y in (0, 1):
        ax.hlines(y, 0, 35 if y else 22, linewidth=.85, color='#b8bfca')
        ax.scatter([2, 8], [y, y], marker='D', color=GREY, s=24, zorder=3)
        ax.scatter([6, 11], [y, y], marker='s', color=TEAL, s=41, zorder=3)
    ax.scatter([0], [1], marker='s', color=RED, s=41, zorder=3)
    ax.scatter([0, 3, 3], [0, 0, 1], marker='x', color=GREY, s=24, zorder=3)
    ax.axvspan(22, 25, color='#c3d4e8', alpha=.6)
    ax.axvspan(28, 36, color='#dbe9e7', alpha=.9)
    ax.vlines(22, -.13, 1.34, color=NAVY, linestyle='--', linewidth=.95)
    ax.text(0, 1.22, 'Evicted\nwrite', ha='center', va='bottom', color=RED, fontsize=8)
    ax.text(3.45, 1.21, 'Absent', ha='center', va='bottom', color=GREY, fontsize=8)
    ax.text(8.5, 1.23, 'Retained writes', ha='center', va='bottom', color=TEAL, fontsize=8)
    ax.text(23.5, 1.39, 'Read', ha='center', fontsize=8)
    ax.text(32, 1.39, 'Output count', ha='center', fontsize=8)
    ax.text(22.32, .08, 'Replace state\nonly if accepted', ha='left', va='bottom', fontsize=8)
    ax.text(5.1, -.15, 'Nonwrite drives (diamonds)', ha='center', va='top', color=GREY, fontsize=8)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#b8bfca')
    fig.savefig(path, metadata=METADATA)
    plt.close(fig)


def effects(path, evidence):
    result = json.loads(evidence.read_text(encoding='utf-8'))
    effects_by_name = {item['effect']: item for item in result['primary_effects']}
    fig, panels = plt.subplots(1, 2, figsize=(7.3, 2.35), layout='constrained')
    for ax, name, title, bounds, ticks in zip(
        panels, ('evicted_gain', 'recent_change'), ('Evicted target', 'Recent record'),
        ((-.32, 3.5), (-2.43, .9)), ([0, 1, 2, 3], [-2, -1, 0]),
    ):
        result = effects_by_name[name]
        values = np.asarray(result['values']) * 100
        lo, hi = np.asarray(result['ci97_5']) * 100
        center = result['mean'] * 100
        ax.axvline(result['threshold'] * 100, color=RED, ls='--', lw=1.15)
        ax.grid(axis='x', color='#e2e7ee', lw=.6)
        ax.errorbar(center, 1.18, xerr=[[max(0., center-lo)], [max(0., hi-center)]], fmt='o',
                    color=TEAL, capsize=4, markersize=6, elinewidth=1.8, zorder=4)
        for value in np.unique(values):
            repeats = np.count_nonzero(values == value)
            ax.scatter(np.full(repeats, value), .2 + .047*np.arange(repeats), color=NAVY, s=11, alpha=.8, zorder=3)
        ax.set(xlim=bounds, ylim=(.03, 1.55), title=title, xlabel='Veto minus old replay (pp)', xticks=ticks)
        ax.set_yticks([.39, 1.18], ['12 blocks', 'Mean and CI'])
        ax.tick_params(axis='y', length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#b8bfca')
    fig.savefig(path, metadata=METADATA)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=HERE.parents[1]/'research'/'dend-tan-2026-11'/'evidence'/'RESULTS.json')
    args = parser.parse_args()
    figures = HERE/'figures'
    figures.mkdir(exist_ok=True)
    schedule(figures/'schedule-and-gate.pdf')
    effects(figures/'primary-effects.pdf', args.evidence)


if __name__ == '__main__':
    main()
