
# A Guide on Text Summarization using NLP techniques

During the execution of my capstone project in the **Machine Learning Engineer Nanodegree in Udacity**, I studied in some depth about the problem of text summarization. For that reason, I am going to write a series of articles about it, from the definition of the problem and some approaches to solve it, showing some basic implementations and algorithms and describing and testing some more advanced techniques. It will take me some posts for the next few weeks or months. 

I will also take advantage of powerful tools like **Amazon SageMaker** containers, **hyperparameter tuning**, **transformers** and **Weights & Biases** logging to show you how to use them to improve and evaluate the performance of the models.

As a summary, some of the future posts will introduce:
- Exploratory Data Analysis for text, to dive deeper in the features of the text and its distribution of words.
- Extractive solutions: Using a simple function from a popular library, gensim, and a Sentence clustering algorithm.
- Abstractive summarization using LSTMs and the attention mechanism
- The Pointer Generation network, an extension from an encoder-decoder, a mix between extractive and abstractive algorithms.
- The Transformer model, extending the attention concept to an initial solution.
- Advanced transformer models like T5 or BART from the fantastic library, transformers by Huggingface.
- Etc,…

## Overview
With the rise of information technologies, globalization and Internet, an enormous amount of information is created daily, including a large volume of written texts. The International Data Corporation (IDC) projects that the total amount of digital data circulating annually around the world would sprout "from 4.4 zettabytes in 2013 to hit 180 zettabytes in 2025" [1]. Dealing with such a huge amount of data is a current problem where automatization techniques can help many industries and businesses.

For example, hundreds or thousands of news are published around the world in a few hours and people do not want to read a full article for ten minutes, So, the development of automatic techniques to get short, concise and understandable summaries would be of great help to many global companies and organizations.

Another use case is social media monitoring, many companies or organizations need to be notified when tweets about their products or brand are mentioned to prepare an appropriate and quick response to them. Other fields of interest are legal contract analysis, question answering and bots, etc.

## Problem Statement
Text Summarization is a challenging problem these days and it can be defined as a technique of shortening a long piece of text to create a coherent and fluent short summary having only the main points in the document.

But, *what is a summary?* It is a *"text that is produced from one or more texts, that contains a significant portion of the information in the original text(s), and that is no longer than half of the original text(s)* [3]. *Summarization clearly involves both these still poorly understood processes, and adds a third (condensation, abstraction, generalization)"*. Or as it is described in [4], text summarization is *"the process of distilling the most important information from a source (or sources) to produce an abridged version for a particular user (or user)and task (or tasks)."*

At this moment, it is a very active field of research and the state-of-the-art solutions are still not so successful than we could expect.

Our main goal in this project is to analyze and build a text summarizer using some basics techniques based on machine learning algorithms. Given a long and descriptive text about some topic we will create a brief and understandable summary covering that same topic. As with any other supervised techniques, we will use a dataset containing pairs of texts and summaries.

## Dataset
When searching for information and data about text summarization I found hard to obtain a "good" dataset. Some of the most popular data sets are intended for research use, containing hundreds of thousands of examples and gigabytes of data that require high computational capacity and days or weeks to train. But we are interested in a dataset that could be trained faster, in a few hours, where we can experiment and develop easily

For that reason, we will use a dataset from Kaggle, called Inshorts News Data. Inshorts is a news service that provides short summaries of news from around the web, scraping the news article from Hindu, Indian times and Guardian. This dataset contains headlines and summary of news items, about 55,000, along with its source.

**Kaggle dataset called Inshorts News Data**, [click this link to access it](https://www.kaggle.com/shashichander009/inshorts-news-data). The datafiles are also included in the **data** directory in this repository.

## Exploratory Data Analysis and preprocess data
**WORK IN PROGRESS**

The Jupyter notebook, **Text_summarization_EDA** describes an EDA on the dataset where we can observe the word and sentence distributions and some other interesting insights. 

It also contains a notebook where we apply some cleaning techniques on text data (dealing with punctuation, stop words,...) and split the data in a train and validation dataset.*Still in progress*

## NEW features in progress

## License
This repository is under the GNU General Public License v3.0.

This repository was developed by Eduardo Muñoz Sala 