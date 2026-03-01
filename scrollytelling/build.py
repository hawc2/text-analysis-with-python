"""Build a 3D scrollytelling essay from a Word2Vec model.

Generates a self-contained HTML file with Three.js + GSAP ScrollTrigger
that narrates a scroll-driven journey through word embedding space.

Usage:
    python build_scrollytelling.py --model word2vec.model --output story.html

    python build_scrollytelling.py --model word2vec.model --top-n 3000 \
        --method tsne --output story_tsne.html
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

TEMPLATE_PATH = Path(__file__).parent / 'template.html'

SECTIONS = [
    {
        'title': 'The Shape of a Genre',
        'body': (
            '<p>You are looking at thousands of words from a corpus of science '
            'fiction magazines published between the 1920s and 1970s, arranged '
            'in three-dimensional space.</p>'
            '<p>Each point is a word. Words that appear in similar contexts '
            'cluster together &mdash; not by spelling or sound, but by '
            '<em>meaning</em>. This is a word embedding: a mathematical '
            'representation of how language works.</p>'
        ),
        'words': [],
        'analogies': [],
        'camera_distance': 100,
        'color': '#6e8898',
    },
    {
        'title': '63,651 Words, 3 Dimensions',
        'body': (
            '<p>The original model places each word in 100-dimensional space. '
            'What you see is a compression &mdash; PCA projects those 100 axes '
            'onto three, preserving the directions of greatest variation.</p>'
            '<p>The most frequent words anchor the center. The interesting '
            'structure lives at the edges, where specialized vocabulary '
            "reveals the genre's preoccupations.</p>"
        ),
        'words': '__TOP_20__',
        'analogies': [],
        'camera_distance': 70,
        'color': '#f0e68c',
    },
    {
        'title': 'Gender in the Galaxy',
        'body': (
            '<p>Even in speculative fiction, language encodes social structures. '
            'The vector from <em>man</em> to <em>woman</em> runs parallel to '
            '<em>king</em> &rarr; <em>queen</em>, <em>hero</em> &rarr; '
            '<em>heroine</em>, <em>husband</em> &rarr; <em>wife</em>.</p>'
            "<p>These parallelisms aren't authored &mdash; they emerge from "
            'patterns of co-occurrence across thousands of stories. The genre '
            'imagined alien civilizations but wrote gender in straight lines.</p>'
        ),
        'words': ['man', 'woman', 'king', 'queen', 'boy', 'girl',
                  'hero', 'heroine', 'husband', 'wife', 'prince', 'princess'],
        'analogies': [
            ('man', 'woman', 'king', 'queen'),
            ('man', 'woman', 'hero', 'heroine'),
        ],
        'camera_distance': 25,
        'color': '#e74c3c',
    },
    {
        'title': 'Bodies and Machines',
        'body': (
            "<p>Science fiction's central metaphor &mdash; the relationship "
            'between human and machine &mdash; is visible as geometric '
            'structure. <em>Man</em> relates to <em>robot</em> as '
            '<em>woman</em> relates to <em>android</em>.</p>'
            "<p>The genre doesn't just tell stories about technology; it "
            'encodes a systematic mapping between organic and artificial.</p>'
        ),
        'words': ['robot', 'android', 'machine', 'human', 'body', 'flesh',
                  'metal', 'brain', 'computer', 'mechanical', 'electronic'],
        'analogies': [
            ('man', 'robot', 'woman', 'android'),
            ('human', 'machine', 'flesh', 'metal'),
        ],
        'camera_distance': 25,
        'color': '#1abc9c',
    },
    {
        'title': 'The Virus Neighborhood',
        'body': (
            '<p>Cluster around <em>virus</em> and you find the vocabulary of '
            'contagion: <em>disease</em>, <em>plague</em>, <em>infection</em>, '
            '<em>parasite</em>. But also <em>cure</em> and <em>serum</em> '
            '&mdash; the genre pairs every threat with its technological '
            'remedy.</p>'
            '<p>In pulp sci-fi, problems exist to be solved.</p>'
        ),
        'words': ['virus', 'disease', 'plague', 'infection', 'cure',
                  'fever', 'blood', 'death', 'parasite', 'serum'],
        'analogies': [
            ('virus', 'cure', 'disease', 'serum'),
        ],
        'camera_distance': 25,
        'color': '#e67e22',
    },
    {
        'title': 'Worlds and Ships',
        'body': (
            '<p>The embedding captures how science fiction constructs its '
            'settings. <em>Ship</em> relates to <em>spaceship</em> the way '
            '<em>car</em> relates to <em>bus</em> &mdash; vehicles differ '
            'by scale.</p>'
            '<p><em>Earth</em> relates to <em>Mars</em> as <em>human</em> '
            'relates to <em>alien</em> &mdash; planets and their inhabitants '
            'are mapped together. The genre builds worlds by analogy.</p>'
        ),
        'words': ['earth', 'mars', 'planet', 'star', 'ship', 'spaceship',
                  'rocket', 'orbit', 'galaxy', 'alien', 'moon'],
        'analogies': [
            ('ship', 'spaceship', 'car', 'bus'),
            ('earth', 'mars', 'human', 'alien'),
        ],
        'camera_distance': 25,
        'color': '#3498db',
    },
    {
        'title': 'Vector Arithmetic',
        'body': (
            '<p>Here is the most remarkable property of embeddings: you can '
            'do algebra with meaning.</p>'
            '<p>Take the vector for <em>king</em>, subtract <em>man</em>, '
            'add <em>woman</em>, and the nearest word to the result is '
            '<em>queen</em>.</p>'
            '<p>The embedding learned that royalty minus masculinity plus '
            'femininity equals a queen &mdash; not from a dictionary, but '
            'from reading science fiction.</p>'
        ),
        'words': ['king', 'queen', 'man', 'woman'],
        'analogies': [],
        'camera_distance': 20,
        'color': '#8e44ad',
        'arithmetic': True,
    },
    {
        'title': 'What Patterns Reveal',
        'body': (
            '<p>Word embeddings make the unconscious patterns of a genre '
            'visible as geometry. The parallel axes, the tight clusters, '
            'the clean analogies &mdash; these are conventions and '
            'assumptions that thousands of authors shared without '
            'coordinating.</p>'
            '<p>A mathematical mirror held up to collective imagination.</p>'
        ),
        'words': [],
        'analogies': [],
        'camera_distance': 100,
        'color': '#ecf0f1',
    },
]


def load_model(model_path):
    model = Word2Vec.load(str(model_path))
    logging.info('Loaded model: %d words, %d dimensions',
                 len(model.wv), model.wv.vector_size)
    return model


def reduce_dimensions(vectors, method, random_state=42):
    if method == 'pca':
        coords = PCA(n_components=3).fit_transform(vectors)
    elif method == 'tsne':
        perplexity = min(30, len(vectors) - 1)
        coords = TSNE(n_components=3, random_state=random_state,
                      perplexity=perplexity).fit_transform(vectors)
    else:
        logging.error('Unknown method: %s', method)
        sys.exit(1)
    # Normalize to [-50, 50] for Three.js
    max_abs = np.max(np.abs(coords))
    if max_abs > 0:
        coords = coords / max_abs * 50
    return coords


def collect_narrative_words(model):
    """Gather all words referenced in narrative sections."""
    narrative = set()
    for section in SECTIONS:
        words = section['words']
        if isinstance(words, list):
            narrative.update(words)
        for analogy in section.get('analogies', []):
            narrative.update(analogy)
    # Filter to vocabulary
    valid = {w for w in narrative if w in model.wv}
    missing = narrative - valid
    if missing:
        logging.warning('Words not in vocabulary: %s', ', '.join(sorted(missing)))
    return valid


def build_data(model, top_n, method):
    """Build the complete JSON data for the scrollytelling template."""
    # Collect words: top-N by frequency + all narrative words
    top_words = model.wv.index_to_key[:top_n]
    narrative_words = collect_narrative_words(model)
    # Merge, preserving frequency order for top words
    all_words = list(top_words)
    for w in sorted(narrative_words):
        if w not in set(all_words):
            all_words.append(w)

    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Reduce to 3D
    logging.info('Reducing %d words to 3D with %s...', len(all_words), method)
    vectors = np.array([model.wv[w] for w in all_words])
    coords = reduce_dimensions(vectors, method)

    # Build section data
    sections_data = []
    for section in SECTIONS:
        # Resolve words
        if section['words'] == '__TOP_20__':
            sec_words = [w for w in model.wv.index_to_key[:20]
                         if w in word_to_idx]
        elif isinstance(section['words'], list):
            sec_words = [w for w in section['words'] if w in word_to_idx]
        else:
            sec_words = []

        indices = [word_to_idx[w] for w in sec_words]

        # Centroid and bounding radius for camera framing
        if indices:
            centroid_arr = coords[indices].mean(axis=0)
            centroid = centroid_arr.tolist()
            dists = np.linalg.norm(coords[indices] - centroid_arr, axis=1)
            bounding_radius = float(np.max(dists))
        else:
            centroid = [0.0, 0.0, 0.0]
            bounding_radius = 50.0

        # Build arrow data
        arrows = []
        for a, b, c, d in section.get('analogies', []):
            if all(w in word_to_idx for w in [a, b, c, d]):
                arrows.append({
                    'from': coords[word_to_idx[a]].tolist(),
                    'to': coords[word_to_idx[b]].tolist(),
                    'from2': coords[word_to_idx[c]].tolist(),
                    'to2': coords[word_to_idx[d]].tolist(),
                    'label': f'{a}\u2192{b} // {c}\u2192{d}',
                })

        # Vector arithmetic data for section 6
        arithmetic = None
        if section.get('arithmetic'):
            arith_words = ['king', 'man', 'woman', 'queen']
            if all(w in word_to_idx for w in arith_words):
                k = coords[word_to_idx['king']]
                m = coords[word_to_idx['man']]
                w = coords[word_to_idx['woman']]
                q = coords[word_to_idx['queen']]
                computed = k + (w - m)
                arithmetic = {
                    'king': k.tolist(),
                    'man': m.tolist(),
                    'woman': w.tolist(),
                    'queen': q.tolist(),
                    'computed': computed.tolist(),
                }

        sections_data.append({
            'title': section['title'],
            'body': section['body'],
            'words': sec_words,
            'indices': indices,
            'centroid': centroid,
            'cameraDistance': section['camera_distance'],
            'boundingRadius': round(bounding_radius, 2),
            'arrows': arrows,
            'color': section['color'],
            'arithmetic': arithmetic,
        })

    return {
        'words': all_words,
        'coords': [[round(float(c), 4) for c in row] for row in coords],
        'sections': sections_data,
        'metadata': {
            'vocabSize': len(model.wv),
            'vectorSize': model.wv.vector_size,
            'nDisplayed': len(all_words),
            'method': method,
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Build a 3D scrollytelling essay from a Word2Vec model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--model', type=Path, required=True,
                   help='Path to Gensim Word2Vec .model file')
    p.add_argument('--top-n', type=int, default=2000,
                   help='Number of top-frequency words for background cloud')
    p.add_argument('--method', choices=['pca', 'tsne'], default='pca',
                   help='Dimensionality reduction method')
    p.add_argument('--output', type=Path, default=Path('scrollytelling.html'),
                   help='Output HTML file')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if not args.model.exists():
        logging.error('Model file not found: %s', args.model)
        sys.exit(2)

    if not TEMPLATE_PATH.exists():
        logging.error('Template not found: %s', TEMPLATE_PATH)
        sys.exit(2)

    model = load_model(args.model)
    data = build_data(model, args.top_n, args.method)

    # Read template and inject data
    template = TEMPLATE_PATH.read_text(encoding='utf-8')
    data_json = json.dumps(data, separators=(',', ':'))
    html = template.replace('{{DATA_PLACEHOLDER}}', data_json)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding='utf-8')
    logging.info('Wrote %s (%.1f KB)', args.output,
                 args.output.stat().st_size / 1024)


if __name__ == '__main__':
    main()
