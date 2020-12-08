
# Capstone Project about Text Summarization using Machine Learning techniques

This repository contains all the documents and code developed for the capstone project in the Machine Learning Engineer Nanodegree program.

The project is about text summarization applying machine learning techniques:

*Text Summarization is a challenging problem these days and it can be defined as a technique of shortening a long piece of text to create a coherent and fluent short summary having only the main points in the document*

## Proposal
The capstone proposal is written in the file name **proposal.pdf** following the rubric defined in the nanodegree program. It contains the sections:
- Domain background
- Problem description and statement
- Dataset and inputs
- Solution Statement
- Benchmark Model
- Evaluation Metric
- Project Design
-  Links

The same content is included in the proposal.md file but with not formatted text.

## Project Report
The main document where we introduce the problem definition, the solution and evaluation of the results is the project report named as **project_report.pdf**. It has been written following a well-organized structure similar to that one described on the [template provided](https://github.com/udacity/machine-learning/blob/master/projects/capstone/capstone_report_template.md).

You can find a detailed description on the models and all the topics of interest in this report, we will not discuss its content in the sections of this README file.

## Dataset
The project is intended to use a **Kaggle dataset called News Summary**, [click this link to access it](https://www.kaggle.com/sunnysai12345/news-summary). The datafiles are also included in the **data** directory in this repository.

The dataset consists in 4515 examples of news and their summaries and some extra data like Author_name, Headlines, Url of Article, Short text, Complete Article. This data was extracted from Inshorts, scraping the news article from Hindu, Indian times and Guardian.
An example:
• Text: "Isha Ghosh, an 81-year-old member of Bharat Scouts and Guides (BSG), has been imparting physical and mental training to schoolchildren ..."
• Summary: "81-yr-old woman conducts physical training in J'khand schools" 

This dataset also include a version with shorter news and summaries, about 98,000 news. They will provide us training and validation data for our abstractive model.

You can download our cleaned dataset in a Kaggle public dataset called [Cleaned News Summary](https://www.kaggle.com/edumunozsala/cleaned-news-summary).
You can also download the Glove embeddings from Kaggle in the folowing dataset [GloVe: Global Vectors for Word Representation](https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation), glove.6B.100d.txt.


## Exploratory Data Analysis and preprocess data

The folder **data_analysis** contains a Jupyter notebook with an EDA on the dataset where we can observe the word and sentence distributions and some other interesting insights. 

It also contains a notebook where we apply some cleaning techniques on text data (dealing with punctuation, stop words,...) and split the data in a train and validation dataset.

## The benchmark model: Gensim Summarizer

This algorithm is included in the folder **gensim_summarizer** of this repository.

*The Gensim summarization module implements TextRank, an unsupervised algorithm based on weighted-graphs from a paper by Mihalcea et al. It is built on top of the popular PageRank algorithm that Google used for ranking.*

In that folder you can find a README which describes the content and how to use it.

## Sentence Embeddings

Our extractive model is based on clustering the sentence embeddings in our source document. The method is described in the report and it is easy to understand and apply.

This model is included in the folder **clustering_summarizer** and a README file will help you to understand it.

## A Sequence-2-Sequence model 
Our first attempt to deal with the summarization problem consists of a sequence-2-sequence model. It is an Encoder-Decoder with attention mechanism that when we analyze the results we could not consider a correct solution. So finally *we discarded this model*

This model is included in the folder **seq2seq_text_summarizer**.

## Pointer Generator Network
In the folder **pointer_generator** you can find an implementation of a encoder-decoder architecture using the pointer generator technique. 

In the README file of that folder we describe the notebook and how to use it. 

## License
This repository is under the GNU General Public License v3.0.

This repository was developed by Eduardo Muñoz Sala 