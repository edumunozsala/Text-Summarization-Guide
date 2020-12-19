import argparse
import json
import sys

import math
import os
import gc
import time
import pandas as pd
import numpy as np
import pickle
import re
import random

import tensorflow as tf

from sklearn.model_selection import train_test_split

# To install packages
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])

# Install the library tensorflow_datasets
install('datasets')
install('rouge_score')
install('wandb')

import datasets
import wandb

from utils import word_tokenize, tensor_to_text, parse_score

from enc_dec_att_model import Encoder, Decoder, BahdanauAttention

def get_train_data(training_dir, train_file, nsamples):

    train_filenamepath = os.path.abspath(os.path.join(training_dir, train_file))
    train_df=pd.read_csv(train_filenamepath, header=0, usecols=[0,1], 
               nrows=nsamples)

    print('Number of train sentences: ',len(train_df))

    input_train = '<start> '+train_df['text'].values+' <end>'
    target_train = '<start> '+train_df['summary'].values+' <end>'

    return input_train, target_train

def get_datasets(encoder_inputs, decoder_outputs, train_frac, batch_size, seed):
    # Split the dataset into a train and validation dataset
    train_enc_inputs, val_enc_inputs, train_dec_outputs, val_dec_outputs, = train_test_split(encoder_inputs, decoder_outputs,
                                                                                       train_size=train_frac,random_state=seed, shuffle=True )
    BUFFER_SIZE = len(train_enc_inputs)
    steps_per_epoch = len(train_enc_inputs)//batch_size
    val_steps_per_epoch = len(val_enc_inputs)//batch_size

    # Define a train dataset 
    dataset = tf.data.Dataset.from_tensor_slices(
                                    (train_enc_inputs, train_dec_outputs))
    dataset = dataset.shuffle(train_enc_inputs.shape[0], reshuffle_each_iteration=True).batch(
                                    batch_size, drop_remainder=True)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Define a train dataset 
    val_dataset = tf.data.Dataset.from_tensor_slices(
                                    (val_enc_inputs, val_dec_outputs))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, val_dataset, steps_per_epoch, val_steps_per_epoch

@tf.function
def train_step(inputs):
# Not using encoder hidden
  inp, targ = inputs
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_states = encoder(inp)

    dec_hidden = [ tf.add(enc_states["for_h"], enc_states["bac_h"]), tf.add(enc_states["for_c"], 
                                enc_states["bac_c"]) ]
    dec_input = tf.expand_dims([tokenizer_outputs.word_index['<start>']] * args.batch_size, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder((dec_input, enc_output, dec_hidden))
      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

@tf.function
def eval_step(inputs):

    inp, targ = inputs
    loss = 0

    enc_output, enc_states = encoder(inp, False)
    dec_hidden = [ tf.add(enc_states["for_h"], enc_states["bac_h"]), tf.add(enc_states["for_c"], 
                          enc_states["bac_c"]) ]
    dec_input = tf.expand_dims([tokenizer_outputs.word_index['<start>']] * args.batch_size, 1)
    result_ids = tf.one_hot([tokenizer_outputs.word_index['<start>']]*args.batch_size, 
                            output_vocab_size, dtype=tf.float32)
    result_ids = tf.expand_dims(result_ids,axis=1)

    #print('Init: ',result_ids.shape,result_ids.dtype)
    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      #decoder(dec_input, enc_output, dec_hidden)
      #predictions, dec_hidden = decoder(dec_input, enc_output, dec_hidden)
      predictions, dec_hidden,_ = decoder((dec_input, enc_output, dec_hidden), False)
      loss += loss_function(targ[:, t], predictions)
      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)
      #
      #print('Predic:',predictions.shape)
      #predicted_ids = tf.argmax(predictions, axis=1)
      #print(predicted_ids.shape, predicted_ids.dtype)
      #print(type(predicted_ids))
      result_ids = tf.concat([result_ids , tf.expand_dims(predictions, axis=1)], 1)
      #t = tf.concat([t,tf.expand_dims(s, axis=1)], axis=1)
          #np.concatenate(([result_ids , predicted_ids ]), axis=1)
      #result_ids = result_ids.T
      #print(result_ids.shape, type(result_ids))

    #result_ids = result_ids.T
    #print('Fin: ',result_ids.shape, type(result_ids))

    batch_loss = (loss / int(targ.shape[1]))
   
    return batch_loss, result_ids

def main_train(dataset, val_dataset, n_epochs, steps_per_epoch, val_steps_per_epoch,
               save_checkpoints=False, logging= False, print_every=50):
  ''' Train the transformer model for n_epochs using the data generator dataset'''
  train_losses = []
  val_losses = []
  val_metric = []

  # In every epoch
  for epoch in range(n_epochs):
    print("Starting epoch {}".format(epoch+1))
    start = time.time()
    # Reset the losss and accuracy calculations
    train_loss.reset_states()
    #Initialize the encoder states
    #enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    #for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    for (batch, dataset_inputs) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(dataset_inputs)
        train_loss.update_state(batch_loss)

        total_loss += batch_loss

        if batch % print_every == 0:
            print('Train: Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
    epoch_loss = total_loss / steps_per_epoch
    train_losses.append(epoch_loss)
    # Register in wandb
    if logging:
        wandb.log({"Training Loss": epoch_loss})

    # Reset the validation losss and accuracy calculations
    val_loss.reset_states()
    total_loss = 0
    # Evaluation loop
    #for (batch, (inp, targ)) in enumerate(val_dataset.take(val_steps_per_epoch)):
    for (batch, dataset_inputs) in enumerate(val_dataset.take(val_steps_per_epoch)):
        _, targ = dataset_inputs
        batch_loss, preds = eval_step(dataset_inputs)#, enc_hidden)
        val_loss.update_state(batch_loss)

        total_loss += batch_loss
        # Add the predictions to the metric
        #Convert sequence to text
        #print(preds.shape, targ.shape)
        preds = tf.argmax(preds, axis=-1)
        #print(preds.shape, preds.dtype)
        predictions = tensor_to_text(tokenizer_outputs, preds.numpy())
        #references = tokenizer_outputs.sequences_to_texts(targ)
        references = tensor_to_text(tokenizer_outputs, targ.numpy())
        metric.add_batch(predictions=predictions, references=references)

    epoch_loss = total_loss / val_steps_per_epoch
    val_losses.append(epoch_loss)
    # compute the metric
    metric_results = metric.compute()
    metric_score = parse_score(metric_results)
    # Save the validation metric
    val_metric.append(metric_score['rouge1'])
    # Register in wandb
    if logging:
        wandb.log({"Validation Loss": val_loss.result(), #})
                   "Rouge1": metric_score['rouge1'],
                   "Rouge2": metric_score['rouge2'],
                   "RougeL": metric_score['rougeL']})

    #Show Validation results
    print("\nValidation: Epoch {} Batch {} Loss {:.4f} Rouge1 {:.4f} Rouge2 {:.4f} RougeL {:.4f}".format(
                epoch+1, batch, epoch_loss,metric_score['rouge1'], metric_score['rouge2'],
                metric_score['rougeL']))

    # Checkpoint the model on every epoch    
    if ((epoch + 1) % 2 == 0) and save_checkpoints:    
      ckpt_save_path = ckpt_manager.save()
      print("Saving checkpoint for epoch {} in {}".format(epoch+1,
                                                        ckpt_save_path))
    print("\nTime for 1 epoch: {} secs\n".format(time.time() - start))

  return train_losses, val_losses, val_metric


def loss_function(target, pred):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_ = loss_object(target, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

if __name__ == '__main__':
    # Install tensorflow_datasets
    #install('tensorflow_datasets')

    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--text-max-len', type=int, default=60, metavar='N',
                        help='input max sequence length for training (default: 60)')
    parser.add_argument('--summ-max-len', type=int, default=15, metavar='N',
                        help='target max sequence length for training (default: 60)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--nsamples', type=int, default=10000, metavar='N',
                        help='number of samples to train (default: 20000)')
    parser.add_argument('--resume', type=bool, default=False, metavar='N',
                        help='Resume training from the latest checkpoint (default: False)')

    # Data parameters                    
    parser.add_argument('--train_file', type=str, default=None, metavar='N',
                        help='Training data file name')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--OOV-token', type=str, default='<unk>', metavar='N',
                        help='Out of vocabulary token ')
    parser.add_argument('--train_frac', type=float, default=0.85, metavar='N',
                        help='Fraction of training data (default: 0.85)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=128, metavar='N',
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--lstm_units', type=int, default=512, metavar='N',
                        help='Units of the Gru layer (default: 512)')
    parser.add_argument('--vocab_size', type=int, default=10000, metavar='N',
                        help='size of the vocabulary (default: 10000)')
    parser.add_argument('--dropout_rate', type=float, default=0.2, metavar='N',
                        help='Dropout Rate for LSTM (default: 0.2)')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='N',
                        help='Learning rate (default: 0.001)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--sm-model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    print(args.sm_model_dir, args.model_dir)
    # Create and config the W&B project
    # Set the project name and run name for wandB
    project_name="Text-summa-EncDec-Attention"
    demo_name="Demo_run"

    # Set the project name, the run name, the description
    wandb.init(project=project_name, name=demo_name, 
                notes="Training en encoder-decoder with attention for Text Summarization")
    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training  
    config = wandb.config          # Initialize config
    config.BATCH_SIZE = args.batch_size    # input batch size for training (default: 64)
    config.EPOCHS = args.epochs        # number of epochs to train (default: 10)
    config.SEED = args.seed               # random seed (default: 42)
    config.MAX_VOCAB_SIZE = args.vocab_size
    config.MAX_SUMM_LENGTH = args.summ_max_len 
    config.MAX_TEXT_LENGTH = args.text_max_len
    config.NUM_SAMPLES = args.nsamples
    config.RNN_UNITS = args.lstm_units
    config.EMBEDDING_DIM = args.embedding_dim
    config.LEARNING_RATE = args.learning_rate

    # Load the training data.
    print("Get the train data")
    input_data, target_data = get_train_data(args.data_dir, args.train_file, args.nsamples)

    # Tokenize and pad the input sequences
    encoder_inputs, tokenizer_inputs, input_max_length = word_tokenize(input_data,args.vocab_size,args.text_max_len, args.OOV_token)
    # Tokenize and pad the outputs sequences
    decoder_outputs, tokenizer_outputs, output_max_length = word_tokenize(target_data,args.vocab_size, args.summ_max_len, args.OOV_token)
    input_vocab_size = tokenizer_inputs.num_words
    output_vocab_size = tokenizer_outputs.num_words

    print('Input vocab: ',input_vocab_size)
    print('Output vocab: ',output_vocab_size)
    
    dataset, val_dataset, steps_per_epoch, val_steps_per_epoch = get_datasets(encoder_inputs, decoder_outputs, args.train_frac, 
                                                                    args.batch_size, args.seed)

    # Create the metric
    metric = datasets.load_metric('rouge')
    
    # Clean the session
    tf.keras.backend.clear_session()
    # Create the Transformer model
    encoder = Encoder(input_vocab_size, args.embedding_dim, args.lstm_units, args.batch_size, args.dropout_rate, None)
    decoder = DecoderBahdanauAtt(output_vocab_size, args.embedding_dim, args.lstm_units, args.batch_size, args.dropout_rate, None)

    # Define a categorical cross entropy loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction="none")
    # Define a metric to store the mean loss of every epoch
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    # Define a matric to save the accuracy in every epoch
    #train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    # Define a metric to store the mean loss of every epoch
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    # Define a matric to save the accuracy in every epoch
    #val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

    # Create the scheduler for learning rate decay
    #leaning_rate = CustomSchedule(D_MODEL)
    # Create the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(args.learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.98,
                                        epsilon=1e-9)
       
    #Create the Checkpoint 
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)

    ckpt_manager = tf.train.CheckpointManager(ckpt, '/opt/ml/checkpoints/', max_to_keep=2)

    if ckpt_manager.latest_checkpoint and args.resume:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Last checkpoint restored.")

    # Train the model
    print('Training the model ....')
    # Train the model
    train_losses, val_losses, val_metric = main_train(dataset, val_dataset, args.epochs, steps_per_epoch, 
                                                  val_steps_per_epoch, save_checkpoints=False, logging= False)
    # Finish the wandb job
    wandb.finish()
    # Save the encoder
    encoder_model_path = os.path.join(args.sm_model_dir, 'encoder')
    tf.saved_model.save(encoder, encoder_model_path)
    # Save the entire model to a HDF5 file
    decoder_model_path = os.path.join(args.sm_model_dir, 'decoder')
    tf.saved_model.save(decoder, decoder_model_path)
    print('Saving the model ....')
    #decoder.save_weights(os.path.join(args.sm_model_dir, 'decoder'), overwrite=True, save_format='tf')

    # Save the parameters used to construct the model
    print("Saving the model parameters")
    model_info_path = os.path.join(args.output_data_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'vocab_size_enc': input_vocab_size,
            'vocab_size_dec': output_vocab_size,
            'embedding_dim': args.embedding_dim,
            'lstm_units': args.lstm_units,
            'batch_size': args.batch_size
        }
        pickle.dump(model_info, f)
          
	# Save the tokenizers with the vocabularies
    print('Saving the dictionaries ....')
    vocabulary_in = os.path.join(args.output_data_dir, 'in_vocab.pkl')
    with open(vocabulary_in, 'wb') as f:
        pickle.dump(tokenizer_inputs, f)

    vocabulary_out = os.path.join(args.output_data_dir, 'out_vocab.pkl')
    with open(vocabulary_out, 'wb') as f:
        pickle.dump(tokenizer_outputs, f)
