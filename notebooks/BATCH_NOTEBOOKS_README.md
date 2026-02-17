# Batch Processing Notebooks

These notebooks provide an easy-to-use interface for running batch text analysis on large datasets.

## Available Notebooks

### 1. [Batch_Topic_Modeling_Runner.ipynb](Batch_Topic_Modeling_Runner.ipynb)

**Purpose:** Train topic models on large datasets with easy configuration

**Use this when:**
- You want to discover topics in a large corpus
- You need to process thousands of documents
- You want a reproducible analysis workflow

**Features:**
- Simple configuration section (just edit variables)
- Runs the full batch processing pipeline
- Shows progress in real-time
- Displays results and visualization directly in the notebook
- Saves configuration for reproducibility

**Quick Start:**
```python
# 1. Open Batch_Topic_Modeling_Runner.ipynb
# 2. Edit Section 1 - Configuration:

INPUT_TYPE = 'directory'  # or 'csv' or 'directory_with_metadata'
INPUT_DIRECTORY = 'Data/my_corpus/'
OUTPUT_DIR = 'results/my_analysis'
NUM_TOPICS = 30

# 3. Run all cells (Cell → Run All)
# 4. View results at the bottom
```

**What you get:**
- Trained topic model
- Interactive visualization
- Topic descriptions
- All model files for reuse

---

### 2. [Apply_Topic_Model_Runner.ipynb](Apply_Topic_Model_Runner.ipynb)

**Purpose:** Apply a trained topic model to new documents

**Use this when:**
- You already have a trained model
- You want to classify new documents
- You need topic assignments for incoming data

**Features:**
- Load existing trained models
- Process new documents
- Get topic assignments with probabilities
- Visualize topic distribution
- Export results to CSV

**Quick Start:**
```python
# 1. Open Apply_Topic_Model_Runner.ipynb
# 2. Edit Section 1 - Configuration:

MODEL_DIR = 'results/topic_modeling_20240216'  # Your trained model directory
INPUT_TYPE = 'csv'
INPUT_CSV = 'Data/new_documents.csv'
OUTPUT_CSV = 'results/topic_assignments.csv'

# 3. Run all cells
# 4. View topic assignments
```

**What you get:**
- CSV with topic assignments for each document
- Topic distribution statistics
- Visualizations
- Summary report

---

## Comparison with Original Notebooks

### Original Notebooks (e.g., Topic_Modeling.ipynb)
- ✅ Great for learning and exploration
- ✅ Interactive and visual
- ✅ Good for small datasets
- ❌ Requires manual execution of each cell
- ❌ Can be slow for large datasets
- ❌ Difficult to reproduce exact settings

### New Batch Processing Notebooks
- ✅ Optimized for large datasets
- ✅ One-click execution (Run All)
- ✅ Configurable and reproducible
- ✅ Progress tracking
- ✅ Automatic result saving
- ✅ Can process 1000s-100,000s of documents
- ✅ Reusable models

**Best approach:** Use original notebooks for learning, use batch notebooks for production analysis

---

## Typical Workflow

### First-time Analysis

1. **Configure and train:**
   - Open [Batch_Topic_Modeling_Runner.ipynb](Batch_Topic_Modeling_Runner.ipynb)
   - Set your data paths and parameters
   - Run all cells
   - Examine the visualization and topics

2. **Experiment:**
   - Try different `NUM_TOPICS` values (10, 20, 30, 50)
   - Adjust `CUSTOM_STOPWORDS` to remove domain-specific terms
   - Change `ALLOWED_POS` to focus on specific word types

3. **Save the best model:**
   - Keep the output directory of your best run
   - Note the configuration used

### Applying to New Data

4. **Classify new documents:**
   - Open [Apply_Topic_Model_Runner.ipynb](Apply_Topic_Model_Runner.ipynb)
   - Point to your trained model directory
   - Specify new documents to classify
   - Run all cells
   - Export topic assignments

---

## Example Configurations

### For a restricted corpus on your local computer

```python
# In Batch_Topic_Modeling_Runner.ipynb

INPUT_TYPE = 'directory'
INPUT_DIRECTORY = '/path/to/restricted/corpus'
FILE_PATTERN = '*.txt'
OUTPUT_DIR = f'results/restricted_analysis_{datetime.now().strftime("%Y%m%d")}'

NUM_TOPICS = 30
PASSES = 25
WORKERS = 4  # Use all CPU cores
SAVE_CORPUS = True  # Save for reuse
```

### For CSV data

```python
INPUT_TYPE = 'csv'
CSV_PATH = 'Data/my_data.csv'
TEXT_COLUMN = 'content'
OUTPUT_DIR = 'results/csv_analysis'

NUM_TOPICS = 20
CUSTOM_STOPWORDS = ['common', 'term', 'specific', 'to', 'domain']
```

### For very large datasets (100,000+ documents)

```python
NUM_TOPICS = 50
PASSES = 30
ITERATIONS = 400
CHUNKSIZE = 500  # Process more at once
WORKERS = 8  # Use more cores
CREATE_VISUALIZATION = False  # Skip for speed
SAVE_CORPUS = True
```

---

## Tips for Best Results

### Getting Good Topics

1. **Start with more topics than you need** - You can always merge similar ones
2. **Remove domain-specific stopwords** - Add common terms to `CUSTOM_STOPWORDS`
3. **Experiment with POS tags** - Try `['NOUN', 'VERB']` for more focused topics
4. **Run multiple times** - Topic modeling has randomness; try a few runs

### Performance Optimization

1. **Use all CPU cores** - Set `WORKERS` to match your CPU
2. **Increase chunk size** - Higher `CHUNKSIZE` = faster but more memory
3. **Skip visualization initially** - Set `CREATE_VISUALIZATION = False` for exploration
4. **Save the corpus** - Set `SAVE_CORPUS = True` to reuse preprocessing

### Reproducibility

1. **Keep configuration in notebook** - It's automatically saved with your run
2. **Set `RANDOM_STATE`** - Same seed = same results
3. **Document custom settings** - Add markdown notes about why you chose parameters
4. **Version your outputs** - Use timestamped directories

---

## Troubleshooting

### "Python not found" error
- Make sure Python is installed and in your PATH
- Try using `python3` instead of `python` in the cells

### "Module not found" error
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Out of memory
- Reduce `CHUNKSIZE`
- Set `CREATE_VISUALIZATION = False`
- Process in batches
- Use smaller spaCy model: `python -m spacy download en_core_web_sm`

### Slow processing
- Increase `WORKERS`
- Reduce `PASSES` and `ITERATIONS` for exploration
- Use smaller spaCy model

### Can't find output files
- Check the output path printed in Section 2
- Look in the `results/` directory
- Check for error messages in the notebook output

---

## Advanced: Batch Processing Multiple Configurations

Create a loop to try multiple parameter combinations:

```python
# Example: Try different topic counts
for num_topics in [10, 20, 30, 50]:
    OUTPUT_DIR = f'results/topics_{num_topics}'
    NUM_TOPICS = num_topics

    # Re-run the processing cells
    # (Copy cells from Sections 3-4)
```

---

## Next Steps

After running topic modeling:

1. **Analyze the visualization** - Open `lda_visualization.html` in browser
2. **Read the topics** - Check `lda_topics.txt` for interpretability
3. **Apply to new data** - Use [Apply_Topic_Model_Runner.ipynb](Apply_Topic_Model_Runner.ipynb)
4. **Integrate into workflow** - Use topic assignments for filtering, classification, etc.

---

## Questions?

- See [BATCH_PROCESSING_GUIDE.md](../BATCH_PROCESSING_GUIDE.md) for detailed documentation
- Check [scripts/README.md](../scripts/README.md) for command-line usage
- Review the original notebooks for algorithm details
