"""Plot the expanded catalog, both with and without repeated vocabularies."""
from collections import defaultdict
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[1]


def group_key(row):
    if row.get('shared_vocabulary_sha256'):
        return 'vocab:' + row['shared_vocabulary_sha256']
    if row.get('tokenizer_artifact_sha256'):
        return 'artifact:' + row['tokenizer_artifact_sha256']
    return 'model:' + row['model_id']


def draw(ax, rows, log_scale, title, minimum, maximum):
    for closed, color, marker in ((False, '#246CA6', 'o'), (True, '#D57127', '^')):
        for proxy in (True, False):
            selected = [r for r in rows if (r['weights_access'] == 'closed_weights') == closed
                        and (r['timeline_date_basis'] != 'documented_release') == proxy]
            if not selected:
                continue
            ax.scatter([datetime.fromisoformat(r['timeline_date']) for r in selected],
                       [r['tokenizer_vocab_size'] for r in selected],
                       s=14 if proxy else 29, marker=marker,
                       facecolors='none' if proxy else color, edgecolors=color,
                       linewidths=0.55, alpha=0.38 if proxy else 0.92,
                       zorder=3 if proxy else 4, rasterized=False)
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(max(1, minimum / 1.5), maximum * 1.35)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(0, maximum * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:g}m' if x >= 1e6 else f'{x/1000:g}k' if x >= 1000 else f'{x:g}'))
    ax.set_xlim(datetime(2017, 11, 1), datetime(2027, 1, 1))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.grid(which='major', color='#E8ECF0', linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold', pad=13)
    ax.set_ylabel('Vocabulary entries' + (' (log scale)' if log_scale else ''), fontsize=10)
    ax.set_xlabel('Documented release / earliest repository record', fontsize=9.5, labelpad=9)
    ax.tick_params(which='major', length=0, pad=7, labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#BAC4CE')
    # Small-vocabulary outliers remain visible; no value-based filtering occurs.
    if log_scale:
        smallest = min(rows, key=lambda r: r['tokenizer_vocab_size'])
        ax.annotate(smallest['model_id'].split('/')[-1][:32],
                    (datetime.fromisoformat(smallest['timeline_date']), smallest['tokenizer_vocab_size']),
                    xytext=(4, 10), textcoords='offset points', fontsize=7, color='#596979')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'workspace/hf-tokenizer-catalog/model-catalog.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'workspace/model-tokenizer-statistics/figures')
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    raw = args.catalog.read_bytes()
    catalog = json.loads(raw)
    collection_date = max(r['collected_at'][:10] for r in catalog if r.get('collected_at'))
    points = [r for r in catalog if r['timeline_date'] and r['tokenizer_vocab_size'] is not None and r['tokenizer_vocab_size'] > 0]
    groups = defaultdict(list)
    for row in points:
        groups[group_key(row)].append(row)
    representatives = [min(group, key=lambda r: (r['timeline_date'], r['model_id'])) for group in groups.values()]
    minimum = min(r['tokenizer_vocab_size'] for r in points)
    maximum = max(r['tokenizer_vocab_size'] for r in points)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'font.size': 10})
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 10.8))
    figure.subplots_adjust(left=0.067, right=0.977, top=0.80, bottom=0.16, wspace=0.20, hspace=0.39)
    for index, data in enumerate((points, representatives)):
        count_label = f'{len(data):,} model rows' if index == 0 else f'{len(data):,} vocabulary/artifact groups'
        draw(axes[index, 0], data, False, f'{chr(97 + 2*index)}) {count_label} · linear y', minimum, maximum)
        draw(axes[index, 1], data, True, f'{chr(98 + 2*index)}) {count_label} · logarithmic y', minimum, maximum)
    figure.suptitle('Vocabulary size across the expanded model catalog', x=0.067, y=0.975,
                   ha='left', fontsize=20, fontweight='bold', color='#1F2D3B')
    figure.text(0.067, 0.934, f'{len(catalog):,} catalog entries  ·  {len(points):,} dated measurements  ·  Collection: {collection_date}',
                fontsize=11, color='#596979')
    legend = [Line2D([], [], linestyle='none', marker='o', color='#246CA6', label='Hugging Face model repositories'),
              Line2D([], [], linestyle='none', marker='^', color='#D57127', label='Closed-model public text encodings'),
              Line2D([], [], linestyle='none', marker='o', markerfacecolor='none', color='#596979', label='Hollow: repository-date proxy')]
    figure.legend(handles=legend, loc='upper left', bbox_to_anchor=(0.061, 0.913), ncol=2,
                  frameon=False, fontsize=10, columnspacing=2, labelspacing=0.9)
    figure.text(0.067, 0.031,
                'No date-range bars and no fitted trend. Repository timestamps do not establish the first public availability of model weights.\n'
                'Bottom row groups identical token-to-ID mappings; formats without that mapping use the primary artifact hash. Dates are earliest observed in this sample.\n'
                'Fine-tunes, merges, conversions, adapters, specialized models, and test fixtures remain in the catalog. Unknown/unavailable vocabularies are omitted.\n'
                'Selection uses current 30-day download rankings, supplemented by curated releases; it is not a historical popularity census.',
                fontsize=8.3, color='#647383', linespacing=1.6)
    for extension in ('png', 'svg'):
        figure.savefig(output / f'vocabulary-catalog.{extension}', dpi=180, facecolor='white')
    plt.close(figure)
    # Individual large panels are easier to read in Chrome than a four-panel thumbnail.
    for name, data in (('all-models', points), ('deduplicated', representatives)):
        for logarithmic, scale in ((False, 'linear'), (True, 'log')):
            figure, ax = plt.subplots(figsize=(12.0, 7.0))
            figure.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.18)
            draw(ax, data, logarithmic, f'{len(data):,} ' + ('model rows' if name == 'all-models' else 'vocabulary/artifact groups'), minimum, maximum)
            figure.suptitle('Tokenizer vocabulary size over time', x=0.08, ha='left', fontsize=19, fontweight='bold')
            figure.text(0.08, 0.91, 'Hollow markers: repository-date proxy. Filled markers: documented release day. No date-range bars.', fontsize=9.5, color='#596979')
            figure.text(0.08, 0.035, 'Blue: Hugging Face repositories. Orange: closed-model public text encodings.\n'
                        'Current-download sample; current tokenizer revisions; repository timestamps are not first-public-access logs.',
                        fontsize=8.5, color='#647383', linespacing=1.5)
            for extension in ('png', 'svg'):
                figure.savefig(output / f'vocabulary-{name}-{scale}.{extension}', dpi=180, facecolor='white')
            plt.close(figure)
    (output / 'expanded-plot-metadata.json').write_text(json.dumps({
        'catalog_rows': len(catalog), 'plotted_model_rows': len(points), 'deduplicated_groups': len(groups),
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'min_vocabulary': minimum, 'max_vocabulary': maximum,
        'date_range_bars': False, 'fitted_trend': None,
        'source_file': str(args.catalog),
        'group_representatives': [{'group': group_key(r), 'model_id': r['model_id'], 'date': r['timeline_date'],
                                   'vocabulary_size': r['tokenizer_vocab_size']} for r in representatives]
    }, indent=2) + '\n')
    print(json.dumps({'catalog_rows': len(catalog), 'plotted_models': len(points), 'groups': len(groups),
                      'vocabulary_range': [minimum, maximum]}))


if __name__ == '__main__':
    main()
