# Batch Processing Guide for Text Analysis

This guide shows you how to convert your Jupyter notebooks into batch processing scripts for running large-scale analyses on your restricted datasets.

## What's Been Created

I've created several tools to help you run batch analyses:

### 1. Main Scripts

- **[scripts/batch_topic_modeling.py](scripts/batch_topic_modeling.py)** - Production-ready topic modeling for large datasets
- **[scripts/apply_topic_model.py](scripts/apply_topic_model.py)** - Apply a trained model to new documents
- **[scripts/convert_notebook_to_script.py](scripts/convert_notebook_to_script.py)** - Convert any notebook to a script
- **[scripts/example_restricted_corpus.sh](scripts/example_restricted_corpus.sh)** - Example shell script

### 2. Documentation

- **[scripts/README.md](scripts/README.md)** - Detailed documentation for all scripts

## Quick Start: Topic Modeling on Your Restricted Corpus

### Step 1: Prepare Your Data

Organize your restricted dataset in one of these formats:

**Option A: Directory of text files**
```
Data/
└── my_restricted_corpus/
    ├── document1.txt
    ├── document2.txt
    └── ...
```

**Option B: CSV file**
```csv
id,content,metadata
1,"text of document 1","..."
2,"text of document 2","..."
```

**Option C: Directory with metadata (like your Gutenberg dataset)**
```
Data/
├── my_corpus/
│   └── clean/
│       ├── doc1.txt
│       └── doc2.txt
└── metadata.csv  (with 'local_path' column)
```

### Step 2: Run Topic Modeling

**For a directory of text files:**
```bash
python scripts/batch_topic_modeling.py \
    --input-dir Data/my_restricted_corpus/ \
    --file-pattern "*.txt" \
    --output-dir results/topic_modeling_$(date +%Y%m%d) \
    --num-topics 30 \
    --passes 25 \
    --workers 4 \
    --verbose
```

**For a CSV file:**
```bash
python scripts/batch_topic_modeling.py \
    --input Data/my_corpus.csv \
    --text-col content \
    --output-dir results/topic_modeling_$(date +%Y%m%d) \
    --num-topics 30 \
    --passes 25 \
    --workers 4 \
    --verbose
```

**For directory with metadata (like Gutenberg):**
```bash
python scripts/batch_topic_modeling.py \
    --input-dir Data/gutenberg-test/clean/ \
    --metadata Data/gutenberg-test/metadata.csv \
    --path-col local_path \
    --output-dir results/gutenberg_topics \
    --num-topics 20 \
    --passes 20
```

### Step 3: View Results

After processing, you'll find these files in your output directory:

- `lda_model` - Saved model (can be reused)
- `lda_dictionary` - Word dictionary
- `lda_topics.txt` - Human-readable topic descriptions
- `lda_topics.json` - Machine-readable topic data
- `lda_visualization.html` - Interactive visualization (open in browser)

**View the visualization:**
```bash
# On Windows
start results/topic_modeling_YYYYMMDD/lda_visualization.html

# On Mac
open results/topic_modeling_YYYYMMDD/lda_visualization.html

# On Linux
xdg-open results/topic_modeling_YYYYMMDD/lda_visualization.html
```

### Step 4: Apply Model to New Documents (Optional)

Once you have a trained model, you can apply it to new documents:

```bash
python scripts/apply_topic_model.py \
    --model results/topic_modeling_YYYYMMDD/lda_model \
    --dictionary results/topic_modeling_YYYYMMDD/lda_dictionary \
    --input new_documents.csv \
    --text-col content \
    --output results/topic_assignments.csv \
    --include-all-topics
```

## Performance Tuning for Large Datasets

### For datasets with 10,000+ documents:

```bash
python scripts/batch_topic_modeling.py \
    --input-dir Data/large_corpus/ \
    --output-dir results/large_analysis \
    --num-topics 50 \
    --passes 30 \
    --iterations 400 \
    --chunksize 500 \
    --workers 8 \
    --no-visualization \
    --save-corpus
```

**Key parameters:**
- `--chunksize 500` - Process 500 docs at a time (increase for more memory)
- `--workers 8` - Use 8 CPU cores (match your CPU)
- `--no-visualization` - Skip HTML generation for speed
- `--save-corpus` - Save preprocessed data for reuse

### For very large datasets (100,000+ documents):

Consider processing in batches:

```bash
# Process first batch
python scripts/batch_topic_modeling.py \
    --input batch1.csv \
    --output-dir results/batch1 \
    --num-topics 50

# Train additional passes on batch 2
# (You'll need to modify the script to support incremental training)
```

## Converting Other Notebooks to Scripts

You can convert any of your other notebooks to batch scripts:

```bash
# Convert a single notebook
python scripts/convert_notebook_to_script.py \
    notebooks/spacy/Spacy-NER-General.ipynb \
    --output scripts/batch_ner.py \
    --clean

# Convert all notebooks in a directory
python scripts/convert_notebook_to_script.py \
    notebooks/ \
    --output-dir scripts/converted/ \
    --clean \
    --recursive
```

The converted scripts will need manual editing to:
1. Add command-line arguments (use [scripts/batch_topic_modeling.py](scripts/batch_topic_modeling.py) as a template)
2. Add batch processing logic
3. Remove any remaining interactive elements

## Common Use Cases

### 1. Topic Modeling Across Different Numbers of Topics

Compare different topic counts to find the optimal number:

```bash
for topics in 10 20 30 50 100; do
    python scripts/batch_topic_modeling.py \
        --input-dir Data/my_corpus/ \
        --output-dir results/topics_${topics} \
        --num-topics $topics \
        --passes 25
done
```

### 2. Adding Custom Stopwords

Remove domain-specific common words:

```bash
python scripts/batch_topic_modeling.py \
    --input Data/my_corpus.csv \
    --text-col content \
    --output-dir results/custom_stops \
    --num-topics 30 \
    --custom-stopwords said went like just really
```

### 3. Processing Only Nouns and Verbs

Focus on core content words:

```bash
python scripts/batch_topic_modeling.py \
    --input-dir Data/my_corpus/ \
    --output-dir results/nouns_verbs \
    --num-topics 30 \
    --allowed-postags NOUN VERB
```

## Troubleshooting

### Memory Issues

If you run out of memory:
- Increase `--chunksize`
- Use `--no-visualization`
- Process in batches
- Close other applications

### Slow Processing

To speed up:
- Use `--workers` matching your CPU cores
- Install a smaller spaCy model: `python -m spacy download en_core_web_sm`
- Reduce `--passes` and `--iterations` for initial exploration

### Installation Issues

Install all dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Or for faster processing with smaller model:
```bash
python -m spacy download en_core_web_sm
```

## Next Steps

1. **Test on a small sample** - Try with 100-1000 documents first
2. **Experiment with parameters** - Adjust `--num-topics`, `--passes`, etc.
3. **Evaluate results** - Use the visualization to assess topic quality
4. **Scale up** - Process your full restricted dataset
5. **Save and reuse models** - Apply trained models to new documents

## Example Workflow for Restricted Dataset

```bash
# 1. Create output directory
mkdir -p results/my_analysis_$(date +%Y%m%d)

# 2. Run topic modeling
python scripts/batch_topic_modeling.py \
    --input-dir /path/to/restricted/corpus \
    --output-dir results/my_analysis_$(date +%Y%m%d) \
    --num-topics 30 \
    --passes 25 \
    --workers 4 \
    --save-corpus \
    --verbose \
    2>&1 | tee results/my_analysis_$(date +%Y%m%d)/log.txt

# 3. Open visualization
# (path will be shown in output)

# 4. If needed, apply model to new documents
python scripts/apply_topic_model.py \
    --model results/my_analysis_YYYYMMDD/lda_model \
    --dictionary results/my_analysis_YYYYMMDD/lda_dictionary \
    --input-dir /path/to/new/documents \
    --output results/my_analysis_YYYYMMDD/new_doc_topics.csv
```

## Additional Resources

- [Gensim LDA Documentation](https://radimrehurek.com/gensim/models/ldamodel.html)
- [spaCy Documentation](https://spacy.io/)
- [pyLDAvis](https://github.com/bmabey/pyLDAvis)
- Original notebooks in [notebooks/](notebooks/)

## Questions?

See the detailed documentation in [scripts/README.md](scripts/README.md) or check the inline help:

```bash
python scripts/batch_topic_modeling.py --help
python scripts/apply_topic_model.py --help
python scripts/convert_notebook_to_script.py --help
```
