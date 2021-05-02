import numpy as np
import re

import os

import tensorflow_datasets as tfds
import tensorflow as tf

#from keras.preprocessing.sequence import pad_sequences

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

# Function to tokenize using Sobwords
def subword_tokenize(corpus, vocab_size, max_length):
  # Create the vocabulary using Subword tokenization
  tokenizer_corpus = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    corpus, target_vocab_size=vocab_size)
  # Get the final vocab size, adding the eos and sos tokens
  num_words = tokenizer_corpus.vocab_size + 2
  # Set eos and sos token
  sos_token = [num_words-2]
  eos_token = [num_words-1]
  # Tokenize the corpus
  sentences = [sos_token + tokenizer_corpus.encode(sentence) + eos_token
          for sentence in corpus]
  # Identify the index of the sentences longer than max length
  idx_to_remove = [count for count, sent in enumerate(sentences)
                 if len(sent) > max_length]
  #Pad the sentences
  sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=max_length)
  
  return sentences, tokenizer_corpus, num_words, sos_token, eos_token, idx_to_remove

def word_tokenize(input_data, max_vocab_size, max_length=None, OOV_token= None, 
                  filters='', padding= True):
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

def tensor_to_text(tokenizer, tensors, eos_token_output):
  texts=[]
  for tensor in tensors:
      text=[]
      for t in tensor:
          if t!=0:
            text +=[tokenizer.index_word[t]]
          if t==eos_token_output:
            break

      #text=[tokenizer.index_word[t] for t in tensor if t!=0 and t!=]
      texts.append(' '.join(text))

  return texts

