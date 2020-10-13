# Text-Analysis-with-Python

This repository contains materials for my Text Analysis with Python workshop for Temple University Libraries' Scholars Studio (Fall 2020)

The workshop will introduce students to Jupyter Notebooks and Google Colab. We will overview the basics of Python for importing and wrangling text data, before turning to more advanced text mining algorithms, topic modeling and vector space modeling (also known as word embeddings).

## Jupyter Notebooks

[Jupyter Notebooks](https://github.com/jupyter/notebook) offer a great way to learn to code and prototype programming applicaitons. Created by [Project Jupyter](https://jupyter.org/), Jupyter Notebooks are open-source, interactive, and web-based, allowing you to easily create, document, and share code and visualizations for many programming languages.

There are a variety of softwares that support the use of Jupyter Notebooks, including integrated development environments like [Anaconda](https://docs.anaconda.com/anaconda/navigator/). Because the installation of programming libraries and the set up of a coding environment can be time-consuming and difficult when dealing with different user set-ups and operating systems, for this workshop, we'll be using Google Colab.

## Google Colab

[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) is a cloud-based, run-time environment for coding in Jupyter Notebooks that takes advantage of Google's powerful computing resources. As long as you have a Google account, using Google Colab can be relatively seamless and user-friendly. Integrated with Google Drive, Google Colab also offers searchable code recommendations to help you develop your code. 

To see examples of Google colab notebooks and what they can do, check out [Awesome Google Colab](https://github.com/firmai/awesome-google-colab). There are plenty of video tutorials on YouTube; I also recommend quick-tip guides like the ones available at [Towards Data Science](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82).

## Text Mining with Python

For a user-friendly tool to learn the basics of text mining, check out [Voyant](https://voyant-tools.org/). 

In the Python programming language, only a few powerful libraries are necessary to achieve interesting results. The [Natural Language Toolkit](https://www.nltk.org) is a reliable library for text mining. The [NLTK Book](https://www.nltk.org/book/) guides you through the basic operations and functions. A [hands-on NLTK tutorial](https://github.com/hb20007/hands-on-nltk-tutorial) has also been created using Jupyter Notebooks.

In this workshop, we'll use NLTK, as well as [Textblob](https://textblob.readthedocs.io/en/dev/), to process and clean text data.

### Topic Modeling

[Topic modeling](http://journalofdigitalhumanities.org/2-1/topic-modeling-a-basic-introduction-by-megan-r-brett/) is one of the most commonly used forms of computational text analysis, helping you identify commonly recurring words and topics as they are distributed across a series of documents. There are many ways to do topic modeling, including multiple different algorithms, and a GUI-based tool by [Scott Enderle](https://github.com/senderle/topic-modeling-tool).

For this workshop, we'll be using a topic modeling algorithm, Latent Dirichlet Allocation (LDA), through the Python library, [Gensim](https://radimrehurek.com/gensim/). The Google Colab notebook I will upload to this repo will walk you through creating an interactive topic modeling visualization using [pyLDAvis](https://github.com/bmabey/pyLDAvis).

### Vector Space Modeling

Vector space modeling (aka word embeddings) is a more recent innovation in text mining, first developed by Google as Word2Vec. Unlike topic modeling, [vector space modeling](http://bookworm.benschmidt.org/posts/2015-10-25-Word-Embeddings.html) allows you to explore the relationships between words in a corpus as they appear in sequence, unlike topic modeling, which typically ignores the order of words in a document. 

For this workshop, we'll be using [Gensim's Word2Vec algorithm](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html). To visualize the word embedding model, we'll be using [Matplotlib](https://matplotlib.org/) and, if the configuration works with Google Colab, the [Tensorflow Embedding Projector](https://projector.tensorflow.org/). For navigating the Tensorflow projector, this [cheatsheet](https://github.com/louishenrifranc/Tensorflow-Cheatsheet) is helpful, although continual changes to the software make many guides out-of-date.

## Sample Datasets

I will provide sample datasets for each workshop session on this repo. For the purposes of the workshop, we won't be using as large a dataset as would normally be ideal. These scripts will work with .txt files and .csv files; you are welcome to use your own datasets if you have them. If you are looking for million word datasets to use for topic modeling or vector space modeling, there are many places to find them, including Google Dataset Search, and Kaggle's datasets. 

Here is a useful dataset of literary and academic corpora at a million words each, available as single text files: http://www.thegrammarlab.com/?nor-portfolio=1000000-word-sample-corpora

If you are looking for something more topical, the Coronavirus Corpus provides a larger set of news articles about the pandemic: https://www.english-corpora.org/corona/
