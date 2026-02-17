"""Apply a trained LDA topic model to new documents.

This script loads a previously trained topic model and applies it to new documents
to get their topic distributions. Useful for classifying or analyzing new data
using an existing topic model.

Usage:
    # Apply model to CSV file
    python scripts/apply_topic_model.py \\
        --model results/lda_model \\
        --dictionary results/lda_dictionary \\
        --input new_documents.csv \\
        --text-col content \\
        --output results/topic_assignments.csv

    # Apply to directory of text files
    python scripts/apply_topic_model.py \\
        --model results/lda_model \\
        --dictionary results/lda_dictionary \\
        --input-dir new_documents/ \\
        --output results/topic_assignments.csv
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pickle

import pandas as pd
import numpy as np
import spacy
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import re
from tqdm import tqdm


def load_bigram_model(model_path: Path):
    """Load bigram model if it exists."""
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


def preprocess_text(
    text: str,
    nlp,
    stop_words: set,
    bigram_mod=None,
    allowed_postags: List[str] = ['NOUN', 'ADJ', 'VERB', 'ADV']
) -> List[str]:
    """Preprocess a single text document."""
    # Clean text
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"'", "", text)

    # Tokenize
    tokens = list(simple_preprocess(text, deacc=True))

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Apply bigrams if available
    if bigram_mod:
        tokens = bigram_mod[tokens]

    # Lemmatize
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc if token.pos_ in allowed_postags]

    return lemmatized


def get_dominant_topic(topic_distribution: List[tuple]) -> Dict[str, Any]:
    """Get the dominant topic from a topic distribution.

    Args:
        topic_distribution: List of (topic_id, probability) tuples

    Returns:
        Dictionary with dominant topic info
    """
    if not topic_distribution:
        return {
            'dominant_topic': -1,
            'topic_prob': 0.0,
            'topic_keywords': ''
        }

    # Sort by probability
    sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    dominant_topic_id, dominant_prob = sorted_topics[0]

    return {
        'dominant_topic': dominant_topic_id,
        'topic_prob': dominant_prob,
    }


def apply_model_to_documents(
    texts: List[str],
    lda_model: LdaModel,
    dictionary: Dictionary,
    nlp,
    stop_words: set,
    bigram_mod=None,
    allowed_postags: List[str] = ['NOUN', 'ADJ', 'VERB', 'ADV'],
    include_all_topics: bool = False
) -> List[Dict[str, Any]]:
    """Apply LDA model to documents and get topic distributions.

    Args:
        texts: List of documents
        lda_model: Trained LDA model
        dictionary: Gensim dictionary
        nlp: spaCy model
        stop_words: Set of stopwords
        bigram_mod: Optional bigram model
        allowed_postags: POS tags to keep
        include_all_topics: Whether to include all topic probabilities

    Returns:
        List of dictionaries with topic assignments
    """
    results = []

    for idx, text in enumerate(tqdm(texts, desc="Processing documents")):
        # Preprocess
        processed = preprocess_text(text, nlp, stop_words, bigram_mod, allowed_postags)

        if not processed:
            results.append({
                'doc_id': idx,
                'dominant_topic': -1,
                'topic_prob': 0.0,
                'num_tokens': 0
            })
            continue

        # Convert to bag of words
        bow = dictionary.doc2bow(processed)

        # Get topic distribution
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)

        # Get dominant topic
        dominant_info = get_dominant_topic(topic_dist)

        result = {
            'doc_id': idx,
            'dominant_topic': dominant_info['dominant_topic'],
            'topic_prob': dominant_info['topic_prob'],
            'num_tokens': len(processed)
        }

        # Include all topic probabilities if requested
        if include_all_topics:
            for topic_id, prob in topic_dist:
                result[f'topic_{topic_id}_prob'] = prob

        results.append(result)

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply trained LDA model to new documents",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model inputs
    parser.add_argument('--model', type=Path, required=True, help='Path to trained LDA model')
    parser.add_argument('--dictionary', type=Path, required=True, help='Path to Gensim dictionary')
    parser.add_argument('--bigram', type=Path, help='Path to bigram model (optional)')

    # Data inputs
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=Path, help='Input CSV file')
    input_group.add_argument('--input-dir', type=Path, help='Input directory with text files')

    parser.add_argument('--text-col', default='content', help='Text column name (for CSV)')
    parser.add_argument('--file-pattern', default='*.txt', help='File pattern (for directory)')

    # Output
    parser.add_argument('--output', type=Path, required=True, help='Output CSV file')
    parser.add_argument('--include-all-topics', action='store_true',
                       help='Include probabilities for all topics (not just dominant)')

    # Processing options
    parser.add_argument('--allowed-postags', nargs='+', default=['NOUN', 'ADJ', 'VERB', 'ADV'],
                       help='POS tags to keep')
    parser.add_argument('--custom-stopwords', nargs='+', help='Additional stopwords')

    args = parser.parse_args(argv)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load models
    logging.info(f"Loading LDA model from {args.model}")
    lda_model = LdaModel.load(str(args.model))

    logging.info(f"Loading dictionary from {args.dictionary}")
    dictionary = Dictionary.load(str(args.dictionary))

    bigram_mod = None
    if args.bigram and args.bigram.exists():
        logging.info(f"Loading bigram model from {args.bigram}")
        bigram_mod = load_bigram_model(args.bigram)

    # Setup NLP tools
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logging.info("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words('english'))
    if args.custom_stopwords:
        stop_words.update(args.custom_stopwords)

    logging.info("Loading spaCy model...")
    try:
        nlp = spacy.load('en_core_web_lg')
    except OSError:
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            logging.error("No spaCy model found. Please install: python -m spacy download en_core_web_lg")
            sys.exit(1)

    # Disable unnecessary pipes
    nlp.disable_pipes([pipe for pipe in nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'lemmatizer']])

    # Load documents
    if args.input:
        logging.info(f"Loading documents from {args.input}")
        df = pd.read_csv(args.input, encoding='utf-8')
        texts = df[args.text_col].astype(str).tolist()
        doc_metadata = df.to_dict('records')
    else:
        logging.info(f"Loading documents from {args.input_dir}")
        files = sorted(args.input_dir.glob(args.file_pattern))
        texts = []
        doc_metadata = []
        for file_path in files:
            try:
                text = file_path.read_text(encoding='utf-8', errors='replace')
                texts.append(text)
                doc_metadata.append({'filename': file_path.name, 'filepath': str(file_path)})
            except Exception as e:
                logging.warning(f"Error reading {file_path}: {e}")

    if not texts:
        logging.error("No documents loaded")
        sys.exit(1)

    logging.info(f"Loaded {len(texts)} documents")

    # Get topic information
    logging.info("Extracting topic keywords...")
    topic_info = {}
    for idx in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(idx, topn=5)
        keywords = ", ".join([word for word, _ in topic_words])
        topic_info[idx] = keywords

    # Apply model
    results = apply_model_to_documents(
        texts,
        lda_model,
        dictionary,
        nlp,
        stop_words,
        bigram_mod,
        args.allowed_postags,
        args.include_all_topics
    )

    # Combine with metadata
    results_df = pd.DataFrame(results)

    # Add original metadata
    metadata_df = pd.DataFrame(doc_metadata)
    results_df = pd.concat([metadata_df, results_df], axis=1)

    # Add topic keywords
    results_df['topic_keywords'] = results_df['dominant_topic'].apply(
        lambda x: topic_info.get(x, '') if x >= 0 else ''
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False, encoding='utf-8')
    logging.info(f"Results saved to {args.output}")

    # Print summary
    logging.info("\n" + "=" * 80)
    logging.info("TOPIC ASSIGNMENT SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total documents: {len(texts)}")
    logging.info(f"Successfully processed: {sum(1 for r in results if r['dominant_topic'] >= 0)}")
    logging.info("\nTopic distribution:")

    topic_counts = results_df[results_df['dominant_topic'] >= 0]['dominant_topic'].value_counts().sort_index()
    for topic_id, count in topic_counts.items():
        keywords = topic_info.get(topic_id, '')
        logging.info(f"  Topic {topic_id} ({keywords}): {count} documents")

    logging.info("=" * 80)


if __name__ == '__main__':
    main()
