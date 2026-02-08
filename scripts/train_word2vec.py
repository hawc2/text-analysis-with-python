"""Train Word2Vec from a CSV corpus (adapted from /nlp/word_embedding_using_word2vec.py).

Usage example:
    python3 train_word2vec.py --input IMDB_Dataset.csv --text-col review \
        --output-model imdb_word2vec.model --vector-size 100

This script is non-Colab, uses gensim 4 APIs, and depends on `nltk` and `pandas`.
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from gensim.models import Word2Vec

from nlp_utils import corpus_from_csv


def train(corpus, vector_size=100, window=5, min_count=2, workers=4, epochs=5):
    model = Word2Vec(sentences=corpus, vector_size=vector_size,
                     window=window, min_count=min_count,
                     workers=workers, epochs=epochs)
    return model


def plot_tsne(model, top_n=500):
    words = model.wv.index_to_key[:top_n]
    wvs = [model.wv[w] for w in words]
    tsne = TSNE(n_components=2, random_state=0)
    pts = tsne.fit_transform(wvs)
    plt.figure(figsize=(12, 8))
    plt.scatter(pts[:, 0], pts[:, 1], s=5)
    for i, w in enumerate(words):
        if i % max(1, top_n // 200) == 0:
            plt.text(pts[i, 0], pts[i, 1], w, fontsize=8)
    plt.title('TSNE of word vectors')
    return plt


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='CSV file path')
    p.add_argument('--text-col', default='review', help='text column name')
    p.add_argument('--output-model', default='word2vec.model', help='output model file')
    p.add_argument('--vector-size', type=int, default=100)
    p.add_argument('--min-count', type=int, default=1)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--tsne', action='store_true', help='produce TSNE plot')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error('Input file not found: %s', input_path)
        sys.exit(2)

    logging.info('Loading corpus from %s', input_path)
    corpus = corpus_from_csv(str(input_path), text_col=args.text_col)
    logging.info('Corpus loaded: %d documents', len(corpus))

    logging.info('Training Word2Vec (size=%d)', args.vector_size)
    model = train(corpus, vector_size=args.vector_size,
                  min_count=args.min_count, epochs=args.epochs)

    out_model = Path(args.output_model)
    model.save(str(out_model))
    logging.info('Model saved to %s', out_model)

    # also save word2vec text format for compatibility
    txt = out_model.with_suffix('.txt')
    model.wv.save_word2vec_format(str(txt), binary=False)
    logging.info('Word2Vec vectors saved to %s', txt)

    if args.tsne:
        logging.info('Computing TSNE (this may be slow)')
        plt = plot_tsne(model)
        plot_file = out_model.with_suffix('.tsne.png')
        plt.savefig(str(plot_file), dpi=150)
        logging.info('TSNE saved to %s', plot_file)


if __name__ == '__main__':
    main()
