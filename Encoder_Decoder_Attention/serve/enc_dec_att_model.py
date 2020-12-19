import numpy as np
import tensorflow as tf


# Create the Encoder based on Bidirectional LSTMs
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, dropout_rate, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.dropout_rate = dropout_rate
        #self.outputs = {}
        self.states = {}
        if embedding_matrix is not None:
            #self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim= embedding_dim, weights = [embedding_matrix])
            self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim= embedding_dim, 
                                                       embeddings_initializer= tf.keras.initializers.Constant(embedding_matrix))
        else:
            self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim= embedding_dim)

        self.lstmb1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units,
                                                                        return_sequences=True,
                                                                        return_state=True,
                                                                        dropout = self.dropout_rate,
                                                                        go_backwards=True), merge_mode='sum')

        self.lstmb2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units,
                                                                        return_sequences=True,
                                                                        return_state= True,
                                                                         dropout=self.dropout_rate, go_backwards=True), merge_mode='sum')
        
        self.lstmb3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units,
                                                                        return_sequences=True,
                                                                        return_state=True,
                                                                        dropout=self.dropout_rate, go_backwards=True), merge_mode='sum')
    
    def call(self, x, training=True):
        x = self.embedding(x)
        #print('x shape: ',x.shape)
        x, hidden_forward, cell_forward, hidden_backward, cell_backward = self.lstmb1(x,
                                                                                      training=training)
        x, hidden_forward, cell_forward, hidden_backward, cell_backward = self.lstmb2(x, initial_state= [hidden_forward, 
                                                                                                        cell_forward, hidden_backward, cell_backward],
                                                                                      training=training)
        outputs, self.states["for_h"], self.states["for_c"], self.states["bac_h"], self.states["bac_c"] = self.lstmb3(x, initial_state=[hidden_forward, 
                                                                                                        cell_forward, hidden_backward, cell_backward], 
                                                                                                        training=training)
        return outputs, self.states
        #return self.outputs, self.states

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)),tf.zeros((self.enc_units, self.enc_units))]

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    #query == states
    #values == output
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


# DEFINING THE DECODER CLASS OF MODEL
class DecoderBahdanauAtt(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, dropout_rate, embedding_matrix):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.dropout_rate = dropout_rate
    if embedding_matrix is not None:
        #self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim= embedding_dim, weights = [embedding_matrix])
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim= embedding_dim, 
                                                   embeddings_initializer= tf.keras.initializers.Constant(embedding_matrix))
    else:
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim= embedding_dim)

    self.lstm1 = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   dropout=self.dropout_rate)

    self.lstm2 = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                  dropout=self.dropout_rate)

    self.lstm3 = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   dropout=self.dropout_rate)

    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)
    

  #def call(self, x, enc_output, hidden, Training=True):
  def call(self, inputs, Training=True):
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x, enc_output, hidden = inputs

    x = self.embedding(x)
    #print('x shape: ',x.shape)

    x, h, c = self.lstm1(x, initial_state = hidden)
    x, h ,c = self.lstm2(x, initial_state= [h, c])
    
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(h, enc_output)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # applying lstm
    x, h, c = self.lstm3(x, initial_state= [h, c])
    states = [h, c]

    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(x)

    return x, states, attention_weights

# DEFINING THE DECODER CLASS OF MODEL
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, dropout_rate, embedding_matrix):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.dropout_rate = dropout_rate
    if embedding_matrix is not None:
        #self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim= embedding_dim, weights = [embedding_matrix])
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim= embedding_dim, 
                                                   embeddings_initializer= tf.keras.initializers.Constant(embedding_matrix))
    else:
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim= embedding_dim)

    self.lstm1 = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   dropout=self.dropout_rate)

    self.lstm2 = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                  dropout=self.dropout_rate)

    self.lstm3 = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   dropout=self.dropout_rate)

    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)
    

  #def call(self, x, enc_output, hidden, Training=True):
  def call(self, inputs, Training=True):
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x, enc_output, hidden = inputs

    x = self.embedding(x)
    #print('x shape: ',x.shape)

    x, h, c = self.lstm1(x, initial_state = hidden)
    x, h ,c = self.lstm2(x, initial_state= [h, c])
    
    # applying lstm
    x, h, c = self.lstm3(x, initial_state= [h, c])
    states = [h, c]

    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(x)

    return x, states

    