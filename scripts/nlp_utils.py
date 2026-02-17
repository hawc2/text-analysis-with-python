"""Small NLP utilities for tokenization and corpus preparation.

This module adapts helpers from the top-level `nlp/` folder for reuse
inside `Hawc2/text-analysis-with-python` scripts.
"""
from typing import List
import csv
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag
from wordcloud import WordCloud
import matplotlib.pyplot as plt

_tokenizer = RegexpTokenizer(r"\w+")
_stopwords = set(stopwords.words('english'))


def tokenize(text: str) -> List[str]:
    """Tokenize a piece of text into word tokens (alpha/numeric).

    Lowercases and returns a list of tokens.
    """
    if not isinstance(text, str):
        return []
    tokens = _tokenizer.tokenize(text)
    return [t.lower() for t in tokens]


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove English stopwords from a token list."""
    return [t for t in tokens if t not in _stopwords]

# --- Noun Extraction and Visualization ---
def extract_nouns(tokens: List[str]) -> List[str]:
    """Extract nouns (NN, NNS) from a list of tokens."""
    tagged_words = pos_tag(tokens)
    return [word for (word, pos) in tagged_words if pos in ['NN', 'NNS']]

def plot_freq_dist(words: List[str], num=20, title="Frequency distribution"):
    """Plot frequency distribution for most common words."""
    freq_dist = nltk.FreqDist(words)
    freq_dist.plot(num, title=title)

def plot_wordcloud(words: List[str], title="Word Cloud", max_font_size=60, colormap='hsv'):
    """Plot a word cloud from a list of words."""
    cloud = WordCloud(max_font_size=max_font_size, colormap=colormap, background_color='white').generate(' '.join(words))
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()

# --- Named Entity Recognition (NER) ---
def extract_entities(text: str) -> List[tuple]:
    """Extract named entities from text using NLTK's ne_chunk."""
    sentences = sent_tokenize(text)
    all_entities = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        tree = nltk.ne_chunk(tagged)
        for subtree in tree:
            if hasattr(subtree, 'label'):
                entity = ' '.join(word for word, tag in subtree.leaves())
                entity_type = subtree.label()
                all_entities.append((entity_type, entity))
    return all_entities

# --- Feature Extraction for Classification ---
def last_letter_feature(word: str) -> dict:
    """Feature: last letter of a word (for gender classification)."""
    return {'last_letter': word[-1]}

def top_words_features(text: str, top_words: List[str]) -> dict:
    """Feature: presence/absence of top words in text."""
    tokens = set(word_tokenize(text))
    return {f'contains({word})': (word in tokens) for word in top_words}

# --- DataFrame Helpers for BookNLP Outputs ---
def character_summary_df(character_data: dict, get_counter_from_dependency_list) -> pd.DataFrame:
    """Create a DataFrame summarizing BookNLP character data."""
    df_list = []
    for character in character_data["characters"]:
        agentList = character["agent"]
        patientList = character["patient"]
        possList = character["poss"]
        modList = character["mod"]
        character_id = character["id"]
        count = character["count"]
        referential_gender = "unknown"
        if character["g"] is not None and character["g"] != "unknown":
            referential_gender = character["g"]["argmax"]
        mentions = character["mentions"]
        max_proper_mention = ""
        if len(mentions["proper"]) > 0:
            max_proper_mention = mentions["proper"][0]["n"]
            df_list.append({
                'Name': max_proper_mention,
                'Character ID': character_id,
                'Mentions': count,
                'Gender': referential_gender,
                'Possessives': get_counter_from_dependency_list(possList).most_common(10),
                'Agent': get_counter_from_dependency_list(agentList).most_common(10),
                'Patient': get_counter_from_dependency_list(patientList).most_common(10),
                'Modifiers': get_counter_from_dependency_list(modList).most_common(10)
            })
    df = pd.DataFrame(df_list)
    df['Character ID'] = df['Character ID'].astype(str)
    return df

def filter_entity_df(entity_df: pd.DataFrame, category: str, top_n=50) -> pd.DataFrame:
    """Filter BookNLP entity DataFrame by category and return top N entities."""
    entity_filter = entity_df['cat'] == category
    return entity_df[entity_filter]['text'].value_counts().reset_index().rename(columns={'index':'entity'})[:top_n]


def normalize_document(text: str) -> List[str]:
    """Full normalize pipeline: tokenize + remove stopwords."""
    return remove_stopwords(tokenize(text))


def corpus_from_csv(path: str, text_col: str = 'review') -> List[List[str]]:
    """Read a CSV and return a list of token lists from `text_col`.

    Uses pandas for convenience; falls back to csv reader if pandas fails.
    """
    try:
        df = pd.read_csv(path, encoding='utf-8')
        texts = df[text_col].astype(str).tolist()
    except Exception:
        texts = []
        with open(path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                texts.append(r.get(text_col, ''))
    return [normalize_document(t) for t in texts]
