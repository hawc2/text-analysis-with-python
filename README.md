# Text-Analysis-with-Python

This repository contains materials for my Text Analysis with Python workshop for Temple University Libraries' Loretta C. Duckworth Scholars Studio (Fall 2020).

The workshop will introduce students to Jupyter Notebooks and Google Colab. We will overview the basics of Python for importing and wrangling text data, before turning to more advanced text mining algorithms, topic modeling and vector space modeling (also known as word embeddings).

## Jupyter Notebooks

[Jupyter Notebooks](https://github.com/jupyter/notebook) offer a great way to learn to code and prototype programming applications. Created by [Project Jupyter](https://jupyter.org/), Jupyter Notebooks are open-source, interactive, and web-based, allowing you to easily create, document, and share code and visualizations for many programming languages.

There are a variety of softwares that support the use of Jupyter Notebooks, including integrated development environments like [Anaconda](https://docs.anaconda.com/anaconda/navigator/). Because the installation of programming libraries and the set up of a coding environment can be time-consuming and difficult when dealing with different user set-ups and operating systems, for this workshop, we'll be using Google Colab.

## Google Colab

[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) is a cloud-based, run-time environment for coding in Jupyter Notebooks that takes advantage of Google's powerful computing resources. As long as you have a Google account, Google Colab can access the files on your Google Drive. It tends to work best with [Google Chrome](https://www.google.com/chrome/index.html). Google Colab also offers useful features like searchable code recommendations to help you develop your scripts. 

To see examples of Google colab notebooks and what they can do, check out [Awesome Google Colab](https://github.com/firmai/awesome-google-colab). There are plenty of video tutorials on YouTube; I also recommend quick-tip guides like the ones available at [Towards Data Science](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82).

## Text Mining with Python

For a user-friendly tool to learn the basics of text mining, check out [Voyant](https://voyant-tools.org/). There's also a book that guides you through using Voyant: check out the online version of [Hermeneutica](http://hermeneuti.ca/). Another cool web-based tool that walks you through cleaning your texts is [Lexos](http://lexos.wheatoncollege.edu/upload).

In the Python programming language, only a few powerful libraries are necessary to achieve interesting results. The [Natural Language Toolkit](https://www.nltk.org) is a reliable library for text mining. The [NLTK Book](https://www.nltk.org/book/) guides you through the basic operations and functions. A [hands-on NLTK tutorial](https://github.com/hb20007/hands-on-nltk-tutorial) has also been created using Jupyter Notebooks.

In this workshop, we'll use a combination of in-built Python functions, NLTK, [Textblob](https://textblob.readthedocs.io/en/dev/), [Spacy](https://spacy.io/), and Gensim to process and clean text data. We also use the Python library, `re` - to learn more about regular expressions, visit https://regexr.com/ and try out the [Regex Crossword](https://regexcrossword.com/).

### Topic Modeling

[Topic modeling](http://journalofdigitalhumanities.org/2-1/topic-modeling-a-basic-introduction-by-megan-r-brett/) is one of the most commonly used forms of computational text analysis, helping you identify commonly recurring words and topics as they are distributed across a series of documents. There are many ways to do topic modeling, including multiple different algorithms, and a GUI-based tool by [Scott Enderle](https://github.com/senderle/topic-modeling-tool).

For this workshop, we'll be using a topic modeling algorithm, Latent Dirichlet Allocation (LDA), through the Python library, [Gensim](https://radimrehurek.com/gensim/). The Google Colab notebook I will upload to this repo will walk you through creating an interactive topic modeling visualization using [pyLDAvis](https://github.com/bmabey/pyLDAvis).

A few tutorials online will guide you through this process in more elaborate detail than the notebooks we'll be using for these workshop sessions. I recommend this toturial from [Machine Learning Plus](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/) and [Towards Data Science](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21).

The Topic Modeling Jupyter Notebook in this repository is prepared to work with two datasets on this repository. One dataset is a collection of Rotten Tomatoes reviews of war films that I scraped using a script on another repo of mine. The other dataset is a collection of extracted features of copyrighte science fiction novels, containing disaggregated bags-of-words sets per chapter for each novel. Warning: the latter dataset won't produce normal bi-grams and tri-grams.

### Vector Space Modeling

Vector space modeling (aka word embeddings) is a more recent innovation in text mining, first developed by Google as Word2Vec. Unlike topic modeling, [vector space modeling](http://bookworm.benschmidt.org/posts/2015-10-25-Word-Embeddings.html) allows you to explore the relationships between words in a corpus as they appear in sequence, unlike topic modeling, which typically ignores the order of words in a document. 

For this workshop, we'll be using [Gensim's Word2Vec algorithm](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html). To visualize the word embedding model, we'll be using [Matplotlib](https://matplotlib.org/) and, if the configuration works with Google Colab, the [Tensorflow Embedding Projector](https://projector.tensorflow.org/). For navigating the Tensorflow projector, this [cheatsheet](https://github.com/louishenrifranc/Tensorflow-Cheatsheet) is helpful, although continual changes to the software make many guides out-of-date.

## Sample Datasets

I will provide sample datasets for each workshop session on this repo. For the purposes of the workshop, we won't be using as large a dataset as would normally be ideal. These scripts will work with .txt files and .csv files; you are welcome to use your own datasets if you have them. If you are looking for million word datasets to use for topic modeling or vector space modeling, there are many places to find them, including Google Dataset Search, and Kaggle's datasets. 

Here is a useful dataset of literary and academic corpora at a million words each, available as [downloadable text files](http://www.thegrammarlab.com/?nor-portfolio=1000000-word-sample-corpora).

If you are looking for something more topical, the Coronavirus Corpus provides a larger set of [news articles about the pandemic](https://www.english-corpora.org/corona/).

If you are interested in how to do text mining on non-English languages, check out [Quinn Dombrowski's Github repo](https://github.com/multilingual-dh/nlp-resources).

## Learning Python 

I recommend learning Python by identifying tasks and projects you want to do *with* Python. Learning Python by developing a [Flask](https://flask.palletsprojects.com/en/1.1.x/) web app, for instance, is a great way to start managing multiple scripts at once. But if you want to start from scratch and learn the basics of Python's syntax and semantics, there are endless tutorials available. 

[Learn Python the Hard Way](https://learntocodetogether.com/learn-python-the-hard-way-free-ebook-download/) is a popular book available freely as a PDF. [The Self-Taught Programmer](https://www.goodreads.com/book/show/51941365-the-self-taught-programmer) is a great introduction to programming using Python. You can also find useful video courses on YouTube, Linked-in Learning, and PluralSight, among others. To use your smartphone to practice programming in Python and other languages, check out [Enki](https://play.google.com/store/apps/details?id=com.enki.insights&hl=en_US&gl=US), [Programming Hub](https://programminghub.io/), or [Mimo](https://getmimo.com/).

If you search around, you can also find plenty of [Intro to Python Jupyter Notebooks](https://jupyter.brynmawr.edu/services/public/dblank/CS245%20Programming%20Languages/2016-Fall/Labs/Chapter%2002%20-%20Introduction%20to%20Python.ipynb).
