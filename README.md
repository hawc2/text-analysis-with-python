# Text-Analysis-with-Python

This repo contains materials relating to my Text Analysis with Python wrkshop for Temple University Libraries' Scholars Studio (Fall 2020)

The workshop, consisting of three-four sessions, will introduce students to Jupyter Notebooks and Google Colab. We will overview the basics of Python for importing and wrangling text data, before turning to more advanced text mining algorithms, topic modeling and vector space modeling (also known as word embeddings).

## Jupyter Notebooks and Google Colab

[Jupyter Notebooks](https://github.com/jupyter/notebook) are a great way to learn to code and prototype programming applicaitons. Creating by [Project Jupyter](https://jupyter.org/), Jupyter Notebooks offering an open-source, interactive web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. 

There are many applications that support the use of Jupyter Notebooks, including integrated development environments like [Anaconda](https://docs.anaconda.com/anaconda/navigator/). Because the installation of packages and set up of a coding environment can be time-consuming and difficult when dealing with different user' set-ups and operating systems, for this workshop, we'll be using Google Colab.

[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) is a cloud-based run-time environment for coding in Jupyter Notebooks that takes advantage of Google's powerful computing resources. If you have a Google account, using Google Colab can be relatively seamless and user-friendly, integrated with Google Drive, and limiting the set-up necessary to start working. Google Colab also offers searchable code chunks to help you develop your code. 

To see examples of Google colab notebooks and what they can do, check out [Awesome Google Colab](https://github.com/firmai/awesome-google-colab).

## Text Mining with Python

For a user-friendly tool to explore the basics of text mining, check out [Voyant](https://voyant-tools.org/). In the Python programming language, only a few powerful libraries are necessary to achieve interesting results. 

The [Natural Language Toolkit](https://www.nltk.org) has been a reliable library for the basics of text mining. The [NLTK Book guides](https://www.nltk.org/book/) you through the fundamental functions. A [hands-on NLTK tutorial](https://github.com/hb20007/hands-on-nltk-tutorial) has been created using Jupyter Notebooks.

In this workshop, we'll be using [Textblob](https://textblob.readthedocs.io/en/dev/), a Python package for processing and cleaning text data that makes use of NLTK, but simplifies some steps in the process. 

### Topic Modeling

[Topic modeling](http://journalofdigitalhumanities.org/2-1/topic-modeling-a-basic-introduction-by-megan-r-brett/) is one of the most commonly used forms of computational text analysis, allowing you to identify a series of topics distributed across a series of documents. There are many ways to do topic modeling, including multiple different algorithms, and a GUI-based tool by [Scott Enderle](https://github.com/senderle/topic-modeling-tool).

For this workshop, we'll be using a topic modeling algorithm, Latent Dirichlet Allocation (LDA), through the Python package, [Gensim](https://radimrehurek.com/gensim/). Using Gensim's standard topic modeling algorithm, the Google Colab notebook in this repo will walk you through creating an interactive topic modeling visualization using [pyLDAvis](https://github.com/bmabey/pyLDAvis).


### Vector Space Modeling

Vector space modeling, also known as word embedding models, is a more recent innovation in text mining, first inaugurated by Google releasing their algorithm, Word2Vec. Unlike topic modeling, [vector space modeling](http://bookworm.benschmidt.org/posts/2015-10-25-Word-Embeddings.html) allows you to explore the relationships between words in the same corpus. 

For this workshop, we'll be using [Gensim's Word2Vec algorithm](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html). To visualize the word embedding model, we'll be using the [Tensorflow Embedding Projector](https://projector.tensorflow.org/). 

## Sample Datasets

For the purposes of the workshop, we won't be using large datasets, simply to save time. If you are looking for million word datasets to use for word embeddings, there are many places to find them, including Google Dataset Search, and Kaggle's datasets. 

Here is a useful dataset of literary and academic corpora at a million words each, in single text files: http://www.thegrammarlab.com/?nor-portfolio=1000000-word-sample-corpora

If you are looking for something more topical, the Coronavirus Corpus provides news documents regarding the pandemic: https://www.english-corpora.org/corona/
