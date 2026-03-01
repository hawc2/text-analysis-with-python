"""Word embedding visualization using the whatlies library.

Load a Gensim Word2Vec model and produce visualizations including arrow
plots, similarity heatmaps, axis projections, and interactive PCA/UMAP
scatter plots. Outputs HTML files (one per visualization type).

Requires: whatlies (pip install whatlies), altair

Usage examples:
    # Arrow plot of word relationships
    python visualize_embeddings_whatlies.py --model word2vec.model \
        --words king queen man woman prince princess --output viz.html

    # Project words onto semantic axes (man-woman on x, king-queen on y)
    python visualize_embeddings_whatlies.py --model word2vec.model \
        --words king queen prince princess duke duchess \
        --x-axis man woman --y-axis rich poor --output projection.html

    # Similarity matrix
    python visualize_embeddings_whatlies.py --model word2vec.model \
        --words dog cat fish bird horse cow \
        --viz-types similarity --output sim.html

    # Multiple visualization types
    python visualize_embeddings_whatlies.py --model word2vec.model \
        --words king queen man woman --viz-types arrow similarity pca \
        --output all_viz.html
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from gensim.models import Word2Vec


def load_model_as_kv(model_path: Path) -> Path:
    """Load Word2Vec model and save KeyedVectors for whatlies."""
    kv_path = model_path.with_suffix('.kv')
    if not kv_path.exists():
        logging.info('Converting model to KeyedVectors format...')
        model = Word2Vec.load(str(model_path))
        model.wv.save(str(kv_path))
        logging.info('Saved KeyedVectors to %s', kv_path)
    else:
        logging.info('Using existing KeyedVectors: %s', kv_path)
    return kv_path


def get_language_and_embeddings(kv_path: Path, words: List[str]):
    """Load GensimLanguage and retrieve embedding set."""
    from whatlies.language import GensimLanguage
    lang = GensimLanguage(str(kv_path))

    # Filter to words actually in vocabulary
    valid = []
    for w in words:
        try:
            lang[w]
            valid.append(w)
        except KeyError:
            logging.warning('Word not in vocabulary: %s', w)

    embset = lang[valid]
    logging.info('Loaded %d embeddings', len(valid))
    return lang, embset


def make_arrow_plot(embset, lang, x_axis_words, y_axis_words):
    """Create arrow plot, optionally projected onto semantic axes."""
    if x_axis_words and y_axis_words:
        x_emb = lang[x_axis_words[0]] - lang[x_axis_words[1]]
        y_emb = lang[y_axis_words[0]] - lang[y_axis_words[1]]
        chart = embset.plot(kind="arrow", x_axis=x_emb, y_axis=y_emb)
    elif x_axis_words:
        x_emb = lang[x_axis_words[0]] - lang[x_axis_words[1]]
        chart = embset.plot(kind="arrow", x_axis=x_emb)
    else:
        chart = embset.plot(kind="arrow")
    return chart


def make_similarity_plot(embset):
    """Create cosine similarity heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    embset.plot_similarity()
    fig = plt.gcf()
    return fig


def make_pca_plot(embset):
    """Create PCA-reduced interactive scatter."""
    from whatlies.transformers import Pca
    return embset.transform(Pca(2)).plot_interactive()


def make_umap_plot(embset):
    """Create UMAP-reduced interactive scatter."""
    try:
        from whatlies.transformers import Umap
    except ImportError:
        logging.error('umap-learn not installed. Run: pip install umap-learn')
        return None
    return embset.transform(Umap(2)).plot_interactive()


def save_chart(chart, output_path: Path, viz_type: str):
    """Save a whatlies/altair/matplotlib chart to file."""
    if chart is None:
        return

    # whatlies returns either altair charts or matplotlib figures
    try:
        # altair chart
        chart.save(str(output_path))
        logging.info('Saved %s visualization to %s', viz_type, output_path)
    except AttributeError:
        # matplotlib figure
        import matplotlib.pyplot as plt
        if hasattr(chart, 'savefig'):
            chart.savefig(str(output_path.with_suffix('.png')), dpi=150,
                          bbox_inches='tight')
        else:
            plt.savefig(str(output_path.with_suffix('.png')), dpi=150,
                        bbox_inches='tight')
        plt.close()
        logging.info('Saved %s visualization to %s', viz_type,
                     output_path.with_suffix('.png'))


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Word embedding visualization using whatlies.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--model', type=Path, required=True,
                   help='Path to Gensim Word2Vec .model file')
    p.add_argument('--words', nargs='+', required=True,
                   help='Words to visualize')
    p.add_argument('--x-axis', nargs=2, default=None, metavar=('W1', 'W2'),
                   help='Two words defining x projection axis (e.g., man woman)')
    p.add_argument('--y-axis', nargs=2, default=None, metavar=('W1', 'W2'),
                   help='Two words defining y projection axis (e.g., rich poor)')
    p.add_argument('--viz-types', nargs='+', default=['arrow'],
                   choices=['arrow', 'similarity', 'pca', 'umap'],
                   help='Visualization types to generate')
    p.add_argument('--output', type=Path, default=Path('whatlies_viz.html'),
                   help='Output file path (suffix added per viz type)')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if not args.model.exists():
        logging.error('Model file not found: %s', args.model)
        sys.exit(2)

    # Collect all words needed (including axis words)
    all_words = list(args.words)
    if args.x_axis:
        all_words.extend(args.x_axis)
    if args.y_axis:
        all_words.extend(args.y_axis)
    all_words = list(dict.fromkeys(all_words))  # dedupe, preserve order

    kv_path = load_model_as_kv(args.model)
    lang, embset = get_language_and_embeddings(kv_path, all_words)

    stem = args.output.stem
    suffix = args.output.suffix or '.html'
    parent = args.output.parent

    makers = {
        'arrow': lambda: make_arrow_plot(embset, lang, args.x_axis, args.y_axis),
        'similarity': lambda: make_similarity_plot(embset),
        'pca': lambda: make_pca_plot(embset),
        'umap': lambda: make_umap_plot(embset),
    }

    for viz_type in args.viz_types:
        logging.info('Generating %s visualization...', viz_type)
        chart = makers[viz_type]()
        if len(args.viz_types) == 1:
            out_path = args.output
        else:
            out_path = parent / f'{stem}_{viz_type}{suffix}'
        save_chart(chart, out_path, viz_type)

    logging.info('Done.')


if __name__ == '__main__':
    main()
