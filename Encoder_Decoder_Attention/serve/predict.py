import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

def predict(sentence, encoder, decoder, max_summ_length, max_text_length, tokenizer_inputs, tokenizer_outputs, 
            input_vocab_size, units):
  attention_plot = np.zeros((max_summ_length, max_text_length))

  sentence = '<start> '+sentence+' <end>'

  inputs = [tokenizer_inputs.word_index[i] if tokenizer_inputs.word_index[i] < input_vocab_size 
            else tokenizer_inputs.word_index['<unk>'] for i in sentence.split(' ')]
  inputs = pad_sequences([inputs], maxlen=max_text_length,padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([tokenizer_outputs.word_index['<start>']], 0)

  for t in range(max_summ_length):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += tokenizer_outputs.index_word[predicted_id] + ' '

    if tokenizer_outputs.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot
