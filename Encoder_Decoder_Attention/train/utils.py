import numpy as np
import re

import os

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(sentences, alpha=False):
    ''' Cleaning process of the text'''
    if alpha:
        # Remove non alphabetic character
        cleaned_text = [''.join([t.lower() for t in text if t.isalpha() or t.isspace()]) for text in sentences]
    else:
        # Simply lower the characters
        cleaned_text = [t.lower() for t in sentences]
    # Remove any emoty string
    cleaned_text = [t for t in cleaned_text if t!='']
    
    return ''.join(cleaned_text)

def one_hot_encode(indices, dict_size):
    ''' Define one hot encode matrix for our sequences'''
    # Creating a multi-dimensional array with the desired output shape
    # Encode every integer with its one hot representation
    features = np.eye(dict_size, dtype=np.float32)[indices.flatten()]
    
    # Finally reshape it to get back to the original array
    features = features.reshape((*indices.shape, dict_size))
            
    return features

def encode_text(input_text, vocab, one_hot = False):
    # Replace every char by its integer value based on the vocabulary
    output = [vocab.get(character,0) for character in input_text]
    
    if one_hot:
    # One hot encode every integer of the sequence
        dict_size = len(vocab)
        return one_hot_encode(output, dict_size)
    else:
        return np.array(output)
# ----------------------------------------------------------------------
# Preprocess the text non breaking the list of words in non_breaking_prefixes
def preprocess_text_nonbreaking(corpus, non_breaking_prefixes):
  corpus_cleaned = corpus
  # Add the string $$$ before the non breaking prefixes
  # To avoid remove dots from some words
  for prefix in non_breaking_prefixes:
    corpus_cleaned = corpus_cleaned.replace(prefix, prefix + '$$$')
  # Remove dots not at the end of a sentence
  corpus_cleaned = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_cleaned)
  # Remove the $$$ mark
  corpus_cleaned = re.sub(r"\.\$\$\$", '', corpus_cleaned)
  # Rmove multiple white spaces
  corpus_cleaned = re.sub(r"  +", " ", corpus_cleaned)

  return corpus_cleaned

def word_tokenize(input_data, max_vocab_size, max_length=None, OOV_token= None, 
                  filters='', padding= True):
    # Create a tokenizer for the input texts and fit it to them 
    tokenizer = Tokenizer(num_words=max_vocab_size, filters=filters, oov_token=OOV_token)
    tokenizer.fit_on_texts(input_data)
    # Tokenize and transform input texts to sequence of integers
    input_sequences = tokenizer.texts_to_sequences(input_data)
    # Claculate the max length
    input_max_len = max(len(s) for s in input_sequences)
    # Apply padding and truncate if required
    if padding:
        if max_length != None:
            input_sequences = pad_sequences(input_sequences, maxlen=max_length, 
                                            truncating='post', padding='post')
        else:
            input_sequences = pad_sequences(input_sequences, padding='post')

    return input_sequences, tokenizer, input_max_len

def convert(tokenizer, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, tokenizer.index_word[t]))

def tensor_to_text(tokenizer, tensors):
  #text=[''.join(tokenizer.index_word[t]) if t!=0
  texts=[]
  tensors = tensors.astype(int)
  for tensor in tensors:
      text=[tokenizer.index_word[t] for t in tensor if t!=0]
      texts.append(' '.join(text))

  return texts

def parse_score(result):
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
