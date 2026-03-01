"""Batch Topic Modeling with LDA using Gensim

This script converts the topic modeling notebook to a batch processing script
suitable for large datasets. It supports both CSV files and directories of text files.

Usage examples:
    # Process CSV file
    python batch_topic_modeling.py --input data/corpus.csv --text-col content \\
        --output-dir results/ --num-topics 20

    # Process directory of text files
    python batch_topic_modeling.py --input-dir Data/gutenberg-test/clean/ \\
        --output-dir results/ --num-topics 20 --file-pattern "*.txt"

    # Process with metadata CSV (for directory input)
    python batch_topic_modeling.py --input-dir Data/gutenberg-test/clean/ \\
        --metadata Data/gutenberg-test/metadata.csv --path-col local_path \\
        --output-dir results/ --num-topics 20

    # Advanced usage with custom parameters
    python batch_topic_modeling.py --input data/corpus.csv --text-col text \\
        --output-dir results/ --num-topics 50 --passes 30 --iterations 400 \\
        --chunksize 500 --workers 4 --no-visualization
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import json
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords
import re
from tqdm import tqdm


class TopicModelingPipeline:
    """Pipeline for batch topic modeling on large datasets."""

    def __init__(
        self,
        num_topics: int = 20,
        passes: int = 20,
        iterations: int = 200,
        chunksize: int = 100,
        random_state: int = 100,
        workers: int = 1,
        allowed_postags: Optional[List[str]] = None,
        custom_stopwords: Optional[List[str]] = None,
        min_count: int = 1,
        bigram_threshold: int = 100,
    ):
        """Initialize the topic modeling pipeline.

        Args:
            num_topics: Number of topics to extract
            passes: Number of passes through the corpus during training
            iterations: Maximum number of iterations through the corpus
            chunksize: Number of documents to process at a time
            random_state: Random seed for reproducibility
            workers: Number of worker threads (for parallel processing)
            allowed_postags: POS tags to keep during lemmatization
            custom_stopwords: Additional stopwords to remove
            min_count: Minimum count for bigrams/trigrams
            bigram_threshold: Threshold for bigram detection
        """
        self.num_topics = num_topics
        self.passes = passes
        self.iterations = iterations
        self.chunksize = chunksize
        self.random_state = random_state
        self.workers = workers
        self.allowed_postags = allowed_postags or ['NOUN', 'ADJ', 'VERB', 'ADV']
        self.min_count = min_count
        self.bigram_threshold = bigram_threshold

        # Download NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logging.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)

        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

        # Load spaCy model
        logging.info("Loading spaCy model...")
        try:
            self.nlp = spacy.load('en_core_web_lg')
        except OSError:
            logging.warning("en_core_web_lg not found, trying en_core_web_sm...")
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logging.error("No spaCy model found. Please install: python -m spacy download en_core_web_lg")
                raise

        # Disable unnecessary spaCy components for speed
        self.nlp.disable_pipes([pipe for pipe in self.nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'morphologizer', 'lemmatizer']])

        self.bigram_mod = None
        self.trigram_mod = None
        self.id2word = None
        self.lda_model = None

    def clean_text(self, text: str) -> str:
        """Clean text by removing emails, extra spaces, and apostrophes."""
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'\S*@\S*\s?', '', text)  # Remove emails
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r"'", "", text)  # Remove apostrophes
        return text

    def sent_to_words(self, texts: List[str]) -> List[List[str]]:
        """Tokenize texts into words."""
        logging.info("Tokenizing texts...")
        for text in tqdm(texts, desc="Tokenizing"):
            yield simple_preprocess(str(text), deacc=True)

    def build_bigrams_trigrams(self, data_words: List[List[str]]):
        """Build bigram and trigram models."""
        logging.info("Building bigram and trigram models...")
        bigram = gensim.models.Phrases(data_words, min_count=self.min_count, threshold=self.bigram_threshold)
        trigram = gensim.models.Phrases(bigram[data_words], threshold=self.bigram_threshold)
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(trigram)

    def remove_stopwords(self, texts: List[List[str]]) -> List[List[str]]:
        """Remove stopwords from texts."""
        logging.info("Removing stopwords...")
        return [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words]
                for doc in tqdm(texts, desc="Removing stopwords")]

    def make_bigrams(self, texts: List[List[str]]) -> List[List[str]]:
        """Create bigrams from texts."""
        logging.info("Creating bigrams...")
        return [self.bigram_mod[doc] for doc in tqdm(texts, desc="Creating bigrams")]

    def lemmatization(self, texts: List[List[str]]) -> List[List[str]]:
        """Lemmatize texts using spaCy."""
        logging.info("Lemmatizing texts (this may take a while)...")
        texts_out = []

        # Process in batches for efficiency
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="Lemmatizing"):
            batch = texts[i:i + batch_size]
            # Join tokens for spaCy processing
            docs = [" ".join(sent) for sent in batch]
            # Process batch through spaCy pipeline
            for doc in self.nlp.pipe(docs, batch_size=50):
                texts_out.append([token.lemma_ for token in doc if token.pos_ in self.allowed_postags])

        return texts_out

    def preprocess(self, texts: List[str]) -> List[List[str]]:
        """Full preprocessing pipeline."""
        logging.info(f"Preprocessing {len(texts)} documents...")

        # Clean texts
        texts = [self.clean_text(text) for text in texts]

        # Tokenize
        data_words = list(self.sent_to_words(texts))

        # Build bigrams/trigrams
        self.build_bigrams_trigrams(data_words)

        # Remove stopwords
        data_words_nostops = self.remove_stopwords(data_words)

        # Make bigrams
        data_words_bigrams = self.make_bigrams(data_words_nostops)

        # Lemmatize
        data_lemmatized = self.lemmatization(data_words_bigrams)

        return data_lemmatized

    def build_corpus(self, texts: List[List[str]]):
        """Build dictionary and corpus."""
        logging.info("Building dictionary and corpus...")
        self.id2word = corpora.Dictionary(texts)
        corpus = [self.id2word.doc2bow(text) for text in tqdm(texts, desc="Building corpus")]
        return corpus

    def train(self, corpus, texts: List[List[str]]):
        """Train the LDA model."""
        logging.info(f"Training LDA model with {self.num_topics} topics...")
        logging.info(f"Parameters: passes={self.passes}, iterations={self.iterations}, chunksize={self.chunksize}")

        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.id2word,
            num_topics=self.num_topics,
            random_state=self.random_state,
            update_every=2,
            chunksize=self.chunksize,
            passes=self.passes,
            iterations=self.iterations,
            alpha='auto',
            per_word_topics=True,
        )

        return self.lda_model

    def save_model(self, output_dir: Path, prefix: str = "lda"):
        """Save the trained model and related artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LDA model
        model_path = output_dir / f"{prefix}_model"
        self.lda_model.save(str(model_path))
        logging.info(f"Model saved to {model_path}")

        # Save dictionary
        dict_path = output_dir / f"{prefix}_dictionary"
        self.id2word.save(str(dict_path))
        logging.info(f"Dictionary saved to {dict_path}")

        # Save bigram/trigram models
        if self.bigram_mod:
            bigram_path = output_dir / f"{prefix}_bigram.pkl"
            with open(bigram_path, 'wb') as f:
                pickle.dump(self.bigram_mod, f)

        # Save topics to text file
        topics_path = output_dir / f"{prefix}_topics.txt"
        with open(topics_path, 'w', encoding='utf-8') as f:
            f.write(f"LDA Model - {self.num_topics} Topics\n")
            f.write("=" * 80 + "\n\n")
            for idx, topic in self.lda_model.print_topics(-1):
                f.write(f"Topic {idx}:\n{topic}\n\n")
        logging.info(f"Topics saved to {topics_path}")

        # Save topics to JSON for programmatic access
        topics_json = output_dir / f"{prefix}_topics.json"
        topics_data = {}
        for idx in range(self.num_topics):
            topic_words = self.lda_model.show_topic(idx, topn=20)
            topics_data[f"topic_{idx}"] = {
                "words": [{"word": word, "probability": float(prob)} for word, prob in topic_words]
            }
        with open(topics_json, 'w', encoding='utf-8') as f:
            json.dump(topics_data, f, indent=2)
        logging.info(f"Topics JSON saved to {topics_json}")

    def create_visualization(self, corpus, output_dir: Path, prefix: str = "lda"):
        """Create pyLDAvis visualization."""
        try:
            import pyLDAvis
            import pyLDAvis.gensim_models

            logging.info("Creating pyLDAvis visualization...")
            vis = pyLDAvis.gensim_models.prepare(self.lda_model, corpus, self.id2word)

            vis_path = output_dir / f"{prefix}_visualization.html"
            pyLDAvis.save_html(vis, str(vis_path))
            logging.info(f"Visualization saved to {vis_path}")
        except ImportError:
            logging.warning("pyLDAvis not installed. Skipping visualization. Install with: pip install pyldavis")
        except Exception as e:
            logging.error(f"Error creating visualization: {e}")


def load_texts_from_csv(csv_path: Path, text_col: str = 'content') -> List[str]:
    """Load texts from a CSV file."""
    logging.info(f"Loading texts from CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV. Available columns: {df.columns.tolist()}")

    texts = df[text_col].astype(str).tolist()
    logging.info(f"Loaded {len(texts)} documents from CSV")
    return texts


def load_texts_from_directory(
    dir_path: Path,
    file_pattern: str = "*.txt",
    metadata_csv: Optional[Path] = None,
    path_col: str = 'local_path'
) -> List[str]:
    """Load texts from a directory of files.

    Args:
        dir_path: Directory containing text files
        file_pattern: Glob pattern for files (e.g., "*.txt")
        metadata_csv: Optional CSV with metadata including file paths
        path_col: Column name in metadata CSV containing file paths
    """
    texts = []

    if metadata_csv and metadata_csv.exists():
        # Load using metadata CSV
        logging.info(f"Loading texts using metadata from {metadata_csv}")
        df = pd.read_csv(metadata_csv, encoding='utf-8')

        if path_col not in df.columns:
            raise ValueError(f"Column '{path_col}' not found in metadata CSV")

        for rel_path in tqdm(df[path_col].fillna(""), desc="Loading files"):
            if not rel_path:
                continue
            text_path = Path(rel_path)
            if not text_path.is_absolute():
                text_path = dir_path / rel_path

            if text_path.exists():
                try:
                    texts.append(text_path.read_text(encoding='utf-8', errors='replace'))
                except Exception as e:
                    logging.warning(f"Error reading {text_path}: {e}")
            else:
                logging.warning(f"File not found: {text_path}")
    else:
        # Load all files matching pattern
        logging.info(f"Loading texts from directory: {dir_path}")
        files = sorted(dir_path.glob(file_pattern))

        if not files:
            raise ValueError(f"No files found matching pattern '{file_pattern}' in {dir_path}")

        for file_path in tqdm(files, desc="Loading files"):
            try:
                texts.append(file_path.read_text(encoding='utf-8', errors='replace'))
            except Exception as e:
                logging.warning(f"Error reading {file_path}: {e}")

    logging.info(f"Loaded {len(texts)} documents from directory")
    return texts


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch Topic Modeling with LDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=Path, help='Input CSV file path')
    input_group.add_argument('--input-dir', type=Path, help='Input directory containing text files')

    parser.add_argument('--text-col', default='content', help='Text column name (for CSV input)')
    parser.add_argument('--file-pattern', default='*.txt', help='File pattern for directory input (default: *.txt)')
    parser.add_argument('--metadata', type=Path, help='Metadata CSV file (for directory input)')
    parser.add_argument('--path-col', default='local_path', help='Path column in metadata CSV')

    # Output options
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory for results')
    parser.add_argument('--prefix', default='lda', help='Prefix for output files (default: lda)')

    # Model parameters
    parser.add_argument('--num-topics', type=int, default=20, help='Number of topics (default: 20)')
    parser.add_argument('--passes', type=int, default=20, help='Number of passes through corpus (default: 20)')
    parser.add_argument('--iterations', type=int, default=200, help='Maximum iterations (default: 200)')
    parser.add_argument('--chunksize', type=int, default=100, help='Chunk size for processing (default: 100)')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker threads (default: 1)')
    parser.add_argument('--random-state', type=int, default=100, help='Random seed (default: 100)')

    # Preprocessing parameters
    parser.add_argument('--min-count', type=int, default=1, help='Minimum count for bigrams (default: 1)')
    parser.add_argument('--bigram-threshold', type=int, default=100, help='Bigram threshold (default: 100)')
    parser.add_argument('--custom-stopwords', nargs='+', help='Additional stopwords to remove')
    parser.add_argument('--allowed-postags', nargs='+', default=['NOUN', 'ADJ', 'VERB', 'ADV'],
                       help='POS tags to keep (default: NOUN ADJ VERB ADV)')

    # Additional options
    parser.add_argument('--no-visualization', action='store_true', help='Skip pyLDAvis visualization')
    parser.add_argument('--save-corpus', action='store_true', help='Save preprocessed corpus')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load texts
    if args.input:
        if not args.input.exists():
            logging.error(f"Input file not found: {args.input}")
            sys.exit(2)
        texts = load_texts_from_csv(args.input, args.text_col)
    else:
        if not args.input_dir.exists():
            logging.error(f"Input directory not found: {args.input_dir}")
            sys.exit(2)
        texts = load_texts_from_directory(
            args.input_dir,
            args.file_pattern,
            args.metadata,
            args.path_col
        )

    if not texts:
        logging.error("No texts loaded. Exiting.")
        sys.exit(1)

    # Initialize pipeline
    pipeline = TopicModelingPipeline(
        num_topics=args.num_topics,
        passes=args.passes,
        iterations=args.iterations,
        chunksize=args.chunksize,
        random_state=args.random_state,
        workers=args.workers,
        allowed_postags=args.allowed_postags,
        custom_stopwords=args.custom_stopwords,
        min_count=args.min_count,
        bigram_threshold=args.bigram_threshold
    )

    # Preprocess
    start_time = datetime.now()
    data_lemmatized = pipeline.preprocess(texts)

    # Build corpus
    corpus = pipeline.build_corpus(data_lemmatized)

    # Save corpus if requested
    if args.save_corpus:
        corpus_path = args.output_dir / f"{args.prefix}_corpus.pkl"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, 'wb') as f:
            pickle.dump(corpus, f)
        logging.info(f"Corpus saved to {corpus_path}")

        texts_path = args.output_dir / f"{args.prefix}_processed_texts.pkl"
        with open(texts_path, 'wb') as f:
            pickle.dump(data_lemmatized, f)
        logging.info(f"Processed texts saved to {texts_path}")

    # Train model
    lda_model = pipeline.train(corpus, data_lemmatized)

    # Save model and results
    pipeline.save_model(args.output_dir, args.prefix)

    # Create visualization
    if not args.no_visualization:
        pipeline.create_visualization(corpus, args.output_dir, args.prefix)

    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logging.info("=" * 80)
    logging.info("TOPIC MODELING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Documents processed: {len(texts)}")
    logging.info(f"Number of topics: {args.num_topics}")
    logging.info(f"Processing time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logging.info(f"Results saved to: {args.output_dir}")
    logging.info("=" * 80)

    # Print top topics
    logging.info("\nTop 5 Topics:")
    for idx in range(min(5, args.num_topics)):
        topic_words = lda_model.show_topic(idx, topn=10)
        words = ", ".join([word for word, _ in topic_words])
        logging.info(f"  Topic {idx}: {words}")


if __name__ == '__main__':
    main()
