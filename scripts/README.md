# Text Analysis Scripts

This directory contains Python scripts for batch processing text analysis tasks on large datasets.

## Available Scripts

### 1. batch_topic_modeling.py

Performs LDA topic modeling on large text corpora. Converts the interactive notebook into a production-ready batch processing script.

**Features:**
- Process CSV files or directories of text files
- Configurable LDA parameters
- Automatic text preprocessing (cleaning, lemmatization, bigrams)
- Saves model, topics, and optional visualization
- Progress tracking with tqdm
- Support for parallel processing

**Basic Usage:**

```bash
# Process CSV file
python scripts/batch_topic_modeling.py \
    --input Data/RottenTomatoes.csv \
    --text-col content \
    --output-dir results/topic_modeling/ \
    --num-topics 20

# Process directory of text files
python scripts/batch_topic_modeling.py \
    --input-dir Data/my_corpus/ \
    --file-pattern "*.txt" \
    --output-dir results/topic_modeling/ \
    --num-topics 30 \
    --passes 30

# Use metadata CSV (like Gutenberg dataset)
python scripts/batch_topic_modeling.py \
    --input-dir Data/gutenberg-test/clean/ \
    --metadata Data/gutenberg-test/metadata.csv \
    --path-col local_path \
    --output-dir results/gutenberg_topics/ \
    --num-topics 20
```

**Advanced Options:**

```bash
# Custom preprocessing and model parameters
python scripts/batch_topic_modeling.py \
    --input Data/corpus.csv \
    --text-col text \
    --output-dir results/advanced/ \
    --num-topics 50 \
    --passes 30 \
    --iterations 400 \
    --chunksize 500 \
    --workers 4 \
    --custom-stopwords movie film review \
    --allowed-postags NOUN VERB \
    --save-corpus \
    --verbose

# Skip visualization (faster for large datasets)
python scripts/batch_topic_modeling.py \
    --input-dir Data/large_corpus/ \
    --output-dir results/large/ \
    --num-topics 100 \
    --no-visualization
```

**Output Files:**
- `lda_model` - Saved Gensim LDA model
- `lda_dictionary` - Word dictionary
- `lda_topics.txt` - Human-readable topic descriptions
- `lda_topics.json` - Machine-readable topic data
- `lda_visualization.html` - Interactive pyLDAvis visualization
- `lda_corpus.pkl` - Preprocessed corpus (if `--save-corpus` is used)

### 2. train_word2vec.py

Train Word2Vec embeddings on text corpora.

**Usage:**

```bash
python scripts/train_word2vec.py \
    --input Data/corpus.csv \
    --text-col review \
    --output-model results/word2vec.model \
    --vector-size 100 \
    --epochs 10
```

### 3. visualize_embeddings_plotly.py

Interactive 3D visualization of word embeddings with analogy arrows using Plotly.

**Features:**
- Load any Gensim Word2Vec model
- PCA, t-SNE, or UMAP reduction to 3D
- Directional arrows showing analogy relationships (e.g., man->woman parallel to king->queen)
- Color-coded semantic groups
- Fully interactive HTML output (rotate, zoom, hover)

**Usage:**

```bash
# Basic: top 200 words in 3D PCA
python scripts/visualize_embeddings_plotly.py \
    --model results/word2vec.model --top-n 200

# With analogy arrows and color groups
python scripts/visualize_embeddings_plotly.py \
    --model results/word2vec.model \
    --words man woman king queen prince princess hero heroine robot android \
    --analogies "man:woman,king:queen;man:woman,hero:heroine" \
    --groups "royalty:king,queen,prince,princess;gender:man,woman;tech:robot,android" \
    --output analogies_3d.html

# Using t-SNE instead of PCA
python scripts/visualize_embeddings_plotly.py \
    --model results/word2vec.model \
    --words ship spaceship car bus truck \
    --method tsne --output transport.html
```

**Output:** Interactive HTML file viewable in any browser.

### 4. visualize_embeddings_whatlies.py

Word embedding visualization using the whatlies library, with arrow plots, semantic axis projections, and similarity heatmaps.

**Features:**
- Arrow plots showing word vector directions
- Project words onto semantic axes (e.g., man-woman axis vs rich-poor axis)
- Cosine similarity heatmaps
- PCA and UMAP interactive scatter plots

**Usage:**

```bash
# Arrow plot
python scripts/visualize_embeddings_whatlies.py \
    --model results/word2vec.model \
    --words king queen man woman prince princess

# Project onto semantic axes
python scripts/visualize_embeddings_whatlies.py \
    --model results/word2vec.model \
    --words king queen prince princess duke duchess \
    --x-axis man woman --y-axis rich poor --output projection.html

# Multiple visualization types
python scripts/visualize_embeddings_whatlies.py \
    --model results/word2vec.model \
    --words king queen man woman prince princess \
    --viz-types arrow similarity pca --output viz.html
```

**Output:** HTML files (one per visualization type if multiple specified).

### 5. scrollytelling/ (separate directory)

3D scrollytelling essay builder. See `scrollytelling/` directory for the build script (`build.py`) and HTML template (`template.html`). Generates scroll-driven 3D narratives through word embedding space using Three.js + GSAP ScrollTrigger.

### 6. nlp_utils.py

Utility functions for text preprocessing, tokenization, and corpus preparation.

## Working with Restricted/Large Datasets

For restricted datasets on your local computer:

1. **Organize your data:**
   ```
   Data/
   └── my_restricted_corpus/
       ├── doc1.txt
       ├── doc2.txt
       └── ...
   ```

2. **Run topic modeling:**
   ```bash
   python scripts/batch_topic_modeling.py \
       --input-dir Data/my_restricted_corpus/ \
       --file-pattern "*.txt" \
       --output-dir results/restricted_analysis/ \
       --num-topics 30 \
       --passes 25 \
       --workers 4
   ```

3. **For very large datasets (millions of documents):**
   - Increase `--chunksize` (e.g., 500 or 1000)
   - Use `--workers` for parallel processing
   - Use `--no-visualization` to skip HTML generation
   - Consider processing in batches if memory is limited

## Performance Tips

1. **Use multiple workers:** `--workers 4` (matches your CPU cores)
2. **Adjust chunksize:** Larger chunks = faster but more memory
3. **Skip visualization:** Use `--no-visualization` for large datasets
4. **Save intermediate results:** Use `--save-corpus` to cache preprocessing
5. **Use smaller spaCy model:** Install `en_core_web_sm` instead of `en_core_web_lg` if speed is more important than accuracy

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Or for faster processing with smaller model:
```bash
python -m spacy download en_core_web_sm
```

## Converting Other Notebooks to Scripts

To convert other notebooks to batch scripts:

1. Remove interactive elements (plots, displays)
2. Add argparse for command-line arguments
3. Add logging instead of print statements
4. Use tqdm for progress bars
5. Save outputs to files instead of displaying
6. Add error handling and validation

See `batch_topic_modeling.py` as a template.
