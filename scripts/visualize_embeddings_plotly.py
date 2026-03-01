"""Interactive 3D visualization of word embeddings using Plotly.

Load a Gensim Word2Vec model and produce an interactive HTML scatter plot
of word vectors reduced to 3 dimensions via PCA, t-SNE, or UMAP.
Optionally draw arrow traces to illustrate analogy relationships.

Usage examples:
    # Basic: top 200 words, PCA reduction
    python visualize_embeddings_plotly.py --model word2vec.model --top-n 200

    # With specific words and t-SNE
    python visualize_embeddings_plotly.py --model word2vec.model \
        --words king queen man woman prince princess --method tsne

    # With analogy arrows showing parallel relationships
    python visualize_embeddings_plotly.py --model word2vec.model \
        --words man woman king queen hero heroine robot android \
        --analogies "man:woman,king:queen;man:woman,hero:heroine" \
        --output analogies.html

    # With semantic groups for color-coding
    python visualize_embeddings_plotly.py --model word2vec.model \
        --groups "royalty:king,queen,prince;gender:man,woman,boy,girl" \
        --method pca --output grouped.html
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.graph_objects as go


GROUP_COLORS = [
    '#e74c3c', '#3498db', '#27ae60', '#e67e22', '#8e44ad',
    '#1abc9c', '#f39c12', '#2c3e50', '#d35400', '#16a085',
]


def load_model(model_path: Path) -> Word2Vec:
    model = Word2Vec.load(str(model_path))
    logging.info('Loaded model: %d words, %d dimensions',
                 len(model.wv), model.wv.vector_size)
    return model


def select_words(model: Word2Vec, words: Optional[List[str]],
                 top_n: int) -> List[str]:
    if words:
        valid = [w for w in words if w in model.wv]
        missing = [w for w in words if w not in model.wv]
        if missing:
            logging.warning('Words not in vocabulary: %s', ', '.join(missing))
        return valid
    return model.wv.index_to_key[:top_n]


def reduce_dimensions(model: Word2Vec, words: List[str],
                      method: str, random_state: int = 42) -> np.ndarray:
    vectors = np.array([model.wv[w] for w in words])
    if method == 'pca':
        return PCA(n_components=3).fit_transform(vectors)
    elif method == 'tsne':
        perplexity = min(30, len(words) - 1)
        return TSNE(n_components=3, random_state=random_state,
                    perplexity=perplexity).fit_transform(vectors)
    elif method == 'umap':
        try:
            from umap import UMAP
        except ImportError:
            logging.error('umap-learn not installed. Run: pip install umap-learn')
            sys.exit(1)
        return UMAP(n_components=3, random_state=random_state).fit_transform(vectors)
    else:
        logging.error('Unknown method: %s', method)
        sys.exit(1)


def parse_analogies(analogy_str: str) -> List[Tuple[str, str, str, str]]:
    """Parse 'a:b,c:d;e:f,g:h' into list of (a,b,c,d) tuples."""
    result = []
    for group in analogy_str.split(';'):
        pairs = group.strip().split(',')
        if len(pairs) != 2:
            logging.warning('Skipping malformed analogy group: %s', group)
            continue
        left = pairs[0].strip().split(':')
        right = pairs[1].strip().split(':')
        if len(left) == 2 and len(right) == 2:
            result.append((left[0], left[1], right[0], right[1]))
        else:
            logging.warning('Skipping malformed analogy: %s', group)
    return result


def parse_groups(groups_str: str) -> Dict[str, List[str]]:
    """Parse 'label1:w1,w2;label2:w3,w4' into dict."""
    result = {}
    for group in groups_str.split(';'):
        parts = group.strip().split(':', 1)
        if len(parts) == 2:
            label = parts[0].strip()
            words = [w.strip() for w in parts[1].split(',')]
            result[label] = words
    return result


def build_figure(words: List[str], coords: np.ndarray,
                 groups: Optional[Dict[str, List[str]]],
                 analogies: List[Tuple[str, str, str, str]],
                 method: str) -> go.Figure:
    word_to_idx = {w: i for i, w in enumerate(words)}
    fig = go.Figure()

    if groups:
        grouped_words = set()
        for i, (label, group_words) in enumerate(groups.items()):
            color = GROUP_COLORS[i % len(GROUP_COLORS)]
            idxs = [word_to_idx[w] for w in group_words if w in word_to_idx]
            grouped_words.update(w for w in group_words if w in word_to_idx)
            if not idxs:
                continue
            fig.add_trace(go.Scatter3d(
                x=coords[idxs, 0], y=coords[idxs, 1], z=coords[idxs, 2],
                mode='markers+text',
                marker=dict(size=6, color=color),
                text=[words[j] for j in idxs],
                textposition='top center',
                textfont=dict(size=10),
                name=label,
                hoverinfo='text',
            ))
        # ungrouped words
        ungrouped = [i for i, w in enumerate(words) if w not in grouped_words]
        if ungrouped:
            fig.add_trace(go.Scatter3d(
                x=coords[ungrouped, 0], y=coords[ungrouped, 1],
                z=coords[ungrouped, 2],
                mode='markers+text',
                marker=dict(size=3, color='#bdc3c7', opacity=0.5),
                text=[words[j] for j in ungrouped],
                textposition='top center',
                textfont=dict(size=7, color='#95a5a6'),
                name='other',
                hoverinfo='text',
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers+text',
            marker=dict(size=4, color='steelblue'),
            text=words,
            textposition='top center',
            textfont=dict(size=8),
            name='words',
            hoverinfo='text',
        ))

    # Analogy arrows
    arrow_colors = ['#e74c3c', '#3498db', '#27ae60', '#e67e22', '#8e44ad',
                    '#1abc9c', '#f39c12', '#d35400']
    for ai, (a, b, c, d) in enumerate(analogies):
        color = arrow_colors[ai % len(arrow_colors)]
        for start, end in [(a, b), (c, d)]:
            if start not in word_to_idx or end not in word_to_idx:
                logging.warning('Skipping arrow %s->%s: word not in plot', start, end)
                continue
            si, ei = word_to_idx[start], word_to_idx[end]
            sx, sy, sz = coords[si]
            ex, ey, ez = coords[ei]
            # Arrow shaft
            fig.add_trace(go.Scatter3d(
                x=[sx, ex], y=[sy, ey], z=[sz, ez],
                mode='lines',
                line=dict(color=color, width=5),
                showlegend=False, hoverinfo='skip',
            ))
            # Arrowhead
            dx, dy, dz = ex - sx, ey - sy, ez - sz
            fig.add_trace(go.Cone(
                x=[ex], y=[ey], z=[ez],
                u=[dx], v=[dy], w=[dz],
                sizemode='absolute', sizeref=0.15,
                anchor='tip',
                colorscale=[[0, color], [1, color]],
                showscale=False, showlegend=False,
                hoverinfo='skip',
            ))
        # Label the analogy pair in the legend
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=0, color=color),
            name=f'{a}:{b} // {c}:{d}',
        ))

    fig.update_layout(
        title=f'Word Embeddings ({method.upper()} 3D)',
        scene=dict(
            xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3',
        ),
        width=1000, height=800,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Interactive 3D word embedding visualization with Plotly.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--model', type=Path, required=True,
                   help='Path to Gensim Word2Vec .model file')
    p.add_argument('--words', nargs='+', default=None,
                   help='Specific words to plot')
    p.add_argument('--top-n', type=int, default=200,
                   help='Number of top words (if --words not given)')
    p.add_argument('--analogies', type=str, default=None,
                   help='Analogy pairs: "a:b,c:d;e:f,g:h"')
    p.add_argument('--groups', type=str, default=None,
                   help='Color groups: "label1:w1,w2;label2:w3,w4"')
    p.add_argument('--method', choices=['pca', 'tsne', 'umap'], default='pca',
                   help='Dimensionality reduction method')
    p.add_argument('--output', type=Path, default=Path('embeddings_3d.html'),
                   help='Output HTML file')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if not args.model.exists():
        logging.error('Model file not found: %s', args.model)
        sys.exit(2)

    model = load_model(args.model)

    # Collect all words needed
    all_words = set(args.words or [])
    analogy_tuples = []
    if args.analogies:
        analogy_tuples = parse_analogies(args.analogies)
        for a, b, c, d in analogy_tuples:
            all_words.update([a, b, c, d])

    group_dict = {}
    if args.groups:
        group_dict = parse_groups(args.groups)
        for ws in group_dict.values():
            all_words.update(ws)

    if all_words:
        word_list = select_words(model, list(all_words), args.top_n)
    else:
        word_list = select_words(model, None, args.top_n)

    logging.info('Reducing %d words to 3D with %s...', len(word_list), args.method)
    coords = reduce_dimensions(model, word_list, args.method)

    fig = build_figure(word_list, coords, group_dict or None,
                       analogy_tuples, args.method)

    fig.write_html(str(args.output))
    logging.info('Saved interactive visualization to %s', args.output)


if __name__ == '__main__':
    main()
