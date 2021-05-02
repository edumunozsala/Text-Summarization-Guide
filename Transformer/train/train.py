import argparse
import json
import sys
#import sagemaker_containers

import math
import os
import gc
import time
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# To install tensorflow_datasets
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])

# Install the library tensorflow_datasets

from utils import subword_tokenize, preprocess_text_nonbreaking, tensor_to_text
#from utils_train import loss_function, CustomSchedule

from model import Transformer

import wandb
# Import datasets library to calculate rouge score
import datasets

INPUT_COLUMN = 'text'
TARGET_COLUMN = 'summary'
OOV_TOKEN='<unk>'

def get_train_data(training_dir, nonbreaking_in, nonbreaking_out, train_file, nsamples):
    # Load the nonbreaking files
    with open(os.path.join(training_dir, nonbreaking_in), 
        mode = "r", encoding = "utf-8") as f:
        non_breaking_prefix_en = f.read()
    with open(os.path.join(training_dir, nonbreaking_out), 
        mode = "r", encoding = "utf-8") as f:
        non_breaking_prefix_es = f.read()

    non_breaking_prefix_en = non_breaking_prefix_en.split("\n")
    non_breaking_prefix_en = [' ' + pref + '.' for pref in non_breaking_prefix_en]
    non_breaking_prefix_es = non_breaking_prefix_es.split("\n")
    non_breaking_prefix_es = [' ' + pref + '.' for pref in non_breaking_prefix_es]
    # Load the training data
    # Load the dataset: sentence in english, sentence in spanish 
    train_filenamepath = os.path.abspath(os.path.join(training_dir, train_file))
    df=pd.read_csv(train_filenamepath, header=0, usecols=[0,1], 
               nrows=nsamples)
    # Preprocess the input data
    input_data=df[INPUT_COLUMN].apply(lambda x : preprocess_text_nonbreaking(x, non_breaking_prefix_en)).tolist()
    # Preprocess and include the end of sentence token to the target text
    target_data=df[TARGET_COLUMN].apply(lambda x : preprocess_text_nonbreaking(x, non_breaking_prefix_es)).tolist()

    input_data = ['<start> '+inp+' <end>' for inp in input_data]
    target_data = ['<start> '+targ+' <end>' for targ in target_data]

    return input_data, target_data

def parse_score(result):
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

@tf.function
def train_step(enc_inputs, dec_inputs, dec_outputs_real):
      with tf.GradientTape() as tape:
          # Call the transformer and get the predicted output
            predictions = transformer(enc_inputs, dec_inputs, True)
            # Calculate the loss
            loss = loss_function(dec_outputs_real, predictions)
      # Update the weights and optimizer
      gradients = tape.gradient(loss, transformer.trainable_variables)
      optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
      # Save and store the metrics
      train_loss.update_state(loss)
      train_accuracy.update_state(dec_outputs_real, predictions)

      return loss

@tf.function
def eval_step(enc_inputs, dec_inputs, dec_outputs_real):
    # Call the transformer and get the predicted output
    predictions = transformer(enc_inputs, dec_inputs, True)
    # Calculate the loss
    loss = loss_function(dec_outputs_real, predictions)
    # Save and store the metrics
    val_loss.update_state(loss)
    val_accuracy.update_state(dec_outputs_real, predictions)

    return loss, predictions

def eval_metrics_step(enc_inputs, dec_inputs, dec_outputs_real):
    # Tokenize the input sequence using the tokenizer_in
    # Set the initial output sentence to sos
    out_sentence = [sos_token_output]*dec_inputs.shape[0]
    out_ids = tf.one_hot([sos_token_output]*dec_inputs.shape[0], 
                            num_words_output, dtype=tf.float32)
    # Reshape the output
    output = tf.expand_dims(out_sentence, axis=1)
    output_ids = tf.expand_dims(out_ids,axis=1)
    #total_loss = 0
    # For max target len tokens
    for _ in range(args.summ_max_len - 1):
        # Call the transformer and get the logits 
        predictions = transformer(enc_inputs, output, False) #(1, seq_length, VOCAB_SIZE_ES)
        #Calculate the loss in the batch
        #loss = loss_function(dec_outputs_real, predictions)
        #total_loss += loss
        # Extract the logists of the next word
        prediction = predictions[:, -1:, :]
        # The highest probability is taken
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        # Check if it is the eos token
        #if predicted_id == eos_token_output:
        #    return tf.squeeze(output, axis=0)
        # Concat the predicted word to the output sequence
        output = tf.concat([output, predicted_id], axis=-1)
        output_ids = tf.concat([output_ids, prediction], axis=1)

    # Calculate the loss
    # output_ids = output_ids[:, 1:, :]
    # Do not consider start token in predictions
    loss = loss_function(dec_outputs_real, output_ids[:, 1:, :])
    # Save and store the metrics
    val_accuracy.update_state(dec_outputs_real, predictions)

    # Call the transformer and get the predicted output
    predictions = output #tf.squeeze(output, axis=0)

    return loss, predictions

def main_train(dataset, val_dataset, transformer, n_epochs, print_every=50):
  ''' Train the transformer model for n_epochs using the data generator dataset'''
  train_losses = []
  train_acc = []
  val_losses = []
  val_acc = []

  # In every epoch
  for epoch in range(n_epochs):
    print("Starting epoch {}".format(epoch+1))
    start = time.time()
    # Reset the losss and accuracy calculations
    train_loss.reset_states()
    train_accuracy.reset_states()
    # Train loop
    # Get a batch of inputs and targets
    for (batch, (enc_inputs, targets)) in enumerate(dataset):
        # Set the decoder inputs
        dec_inputs = targets[:, :-1]
        # Set the target outputs, right shifted
        dec_outputs_real = targets[:, 1:]
        # Apply a step train
        loss = train_step(enc_inputs, dec_inputs, dec_outputs_real)

        if batch % print_every == 0:
            train_losses.append(train_loss.result())
            train_acc.append(train_accuracy.result())
            # Register in wandb
            wandb.log({"Training Loss": train_loss.result()})

            print('Train: Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   train_loss.result()))

    # Reset the validation losss and accuracy calculations
    val_loss.reset_states()
    val_accuracy.reset_states()
    # Evaluation loop
    total_loss=0
    i=0
    for (batch, (enc_inputs, targets)) in enumerate(val_dataset):
        i+=1
        # Set the decoder inputs
        dec_inputs = targets[:, :-1]
        # Set the target outputs, right shifted
        dec_outputs_real = targets[:, 1:]
        # Apply a step train
        loss, preds = eval_metrics_step(enc_inputs, dec_inputs, dec_outputs_real)
        total_loss += loss
        # The highest probability is taken
        predicted_sentences = tensor_to_text(tokenizer_outputs, preds.numpy(),eos_token_output)                
        # TRansform 
        references = tensor_to_text(tokenizer_outputs, targets.numpy(),eos_token_output)
        metric.add_batch(predictions=predicted_sentences, references=references)
    #print('i: ',i)
    #print('batch: ',batch)
    total_loss = total_loss/i
    val_loss.update_state(total_loss)
    
    val_losses.append(val_loss.result())
    val_acc.append(val_accuracy.result())
    # Calculate the rouge score for the epoch
    metric_results = metric.compute()
    metric_score = parse_score(metric_results)

    # Register in wandb
    wandb.log({"Validation Loss": val_loss.result(), #})
                   "Rouge1": metric_score['rouge1'],
                   "Rouge2": metric_score['rouge2'],
                   "RougeL": metric_score['rougeL']})
    
    #Show Validation results
    print("\nValidation: Epoch {} Batch {} Loss {:.4f} Rouge1 {:.4f} Rouge2 {:.4f} RougeL {:.4f}".format(
                epoch+1, batch, val_loss.result(),metric_score['rouge1'], metric_score['rouge2'],
                metric_score['rougeL']))

    # Checkpoint the model on every epoch        
    ckpt_save_path = ckpt_manager.save()
    print("Saving checkpoint for epoch {} in {}".format(epoch+1,
                                                        ckpt_save_path))
    print("Time for 1 epoch: {} secs\n".format(time.time() - start))

  return train_losses, train_acc, val_losses, val_acc

def loss_function(target, pred):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_ = loss_object(target, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--text_max-len', type=int, default=65, metavar='N',
                        help='input max sequence length for training (default: 65)')
    parser.add_argument('--summ_max-len', type=int, default=15, metavar='N',
                        help='Output max sequence length for training (default: 15)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--nsamples', type=int, default=10000, metavar='N',
                        help='number of samples to train (default: 20000)')
    parser.add_argument('--resume', type=bool, default=False, metavar='N',
                        help='Resume training from the latest checkpoint (default: False)')

    # Data parameters                    
    parser.add_argument('--train_file', type=str, default=None, metavar='N',
                        help='Training data file name')
    parser.add_argument('--non_breaking_in', type=str, default=None, metavar='N',
                        help='Non breaking prefixes for input vocabulary')
    parser.add_argument('--non_breaking_out', type=str, default=None, metavar='N',
                        help='Non breaking prefixes for output vocabulary')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--train_frac', type=float, default=0.85, metavar='N',
                        help='Fraction of training data (default: 0.85)')


    # Model Parameters
    parser.add_argument('--d_model', type=int, default=128, metavar='N',
                        help='Model dimension (default: 64)')
    parser.add_argument('--ffn_dim', type=int, default=128, metavar='N',
                        help='size of the FFN layer (default: 128)')
    parser.add_argument('--vocab_size', type=int, default=16384, metavar='N',
                        help='size of the vocabulary (default: 10000)')
    parser.add_argument('--n_layers', type=int, default=6, metavar='N',
                        help='number of layers (default: 4)')
    parser.add_argument('--n_heads', type=int, default=8, metavar='N',
                        help='number of heads (default: 8)')
    parser.add_argument('--dropout_rate', type=float, default=0.1, metavar='N',
                        help='Dropout rate (default: 0.1)')

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
    project_name="transformer_demo"
    demo_name="Transformer-Tuner"
    group_name = "Transformer"

    if args.resume:
        # Set the project name, the run name, the description
        wandb.init(project=project_name, name=demo_name, group=group_name,
                    notes="Training a Transformer model for Text Summarization")
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
        config.D_MODEL = args.d_model
        config.FFN_UNITS = args.ffn_dim
        config.DROPOUT_RATE = args.dropout_rate
        config.N_LAYERS = args.n_layers
        config.N_HEADS = args.n_heads

    # Load the training data.
    print("Get the train data")
    input_data, target_data = get_train_data(args.data_dir, args.non_breaking_in, args.non_breaking_out, args.train_file, args.nsamples)

    tokenizer = Tokenizer(num_words=args.vocab_size, filters='', oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(input_data)
    tokenizer.fit_on_texts(target_data)
    # Tokenize and transform input texts to sequence of integers
    input_sequences = tokenizer.texts_to_sequences(input_data)
    output_sequences = tokenizer.texts_to_sequences(target_data)
    # Claculate the max length
    input_max_length = max(len(s) for s in input_sequences)
    output_max_length = max(len(s) for s in output_sequences)
    # Apply padding and truncate if required
    encoder_inputs = pad_sequences(input_sequences, maxlen=args.text_max_len, 
                                            truncating='post', padding='post')
    decoder_outputs = pad_sequences(output_sequences, maxlen=args.summ_max_len, 
                                            truncating='post', padding='post')
    tokenizer_inputs = tokenizer
    tokenizer_outputs = tokenizer

    #Set the start and end token for input and output vocabulary
    sos_token_input = tokenizer_inputs.word_index['<start>']
    eos_token_input = tokenizer_inputs.word_index['<end>']
    sos_token_output = tokenizer_outputs.word_index['<start>']
    eos_token_output = tokenizer_outputs.word_index['<end>']
    print('Token for sos and eos:', sos_token_input, eos_token_input, sos_token_output, eos_token_output)
    num_words_inputs = tokenizer_inputs.num_words
    num_words_output = tokenizer_outputs.num_words
    print('Size of Input Vocabulary: ', num_words_inputs)
    print('Size of Output Vocabulary: ', num_words_output)

    print('Input vocab: ',num_words_inputs)
    print('Output vocab: ',num_words_output)

    # Split the dataset into train and validation
    train_enc_inputs, val_enc_inputs, train_dec_outputs, val_dec_outputs, = train_test_split(encoder_inputs, decoder_outputs,
                                                                                       train_size=args.train_frac,random_state=args.seed, shuffle=True )
    # Define a train dataset 
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_enc_inputs, train_dec_outputs))
    dataset = dataset.shuffle(train_enc_inputs.shape[0], reshuffle_each_iteration=True).batch(
        args.batch_size, drop_remainder=True)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Define a train dataset 
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_enc_inputs, val_dec_outputs))
    val_dataset = val_dataset.batch(args.batch_size, drop_remainder=True)

    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Create the metric
    metric = datasets.load_metric('rouge')

    # Clean the session
    tf.keras.backend.clear_session()
    # Create the Transformer model
    transformer = Transformer(vocab_size_enc=num_words_inputs,
                          vocab_size_dec=num_words_output,
                          d_model=args.d_model,
                          n_layers=args.n_layers,
                          FFN_units=args.ffn_dim,
                          n_heads=args.n_heads,
                          dropout_rate=args.dropout_rate)

    # Define a categorical cross entropy loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction="none")
    # Define a metric to store the mean loss of every epoch
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    # Define a matric to save the accuracy in every epoch
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    # Define a metric to store the mean loss of every epoch
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    # Define a matric to save the accuracy in every epoch
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

    # Create the scheduler for learning rate decay
    leaning_rate = CustomSchedule(args.d_model)
    # Create the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(leaning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

    #Create the Checkpoint 
    print('Creating the checkpoint ...')
    ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, '/opt/ml/checkpoints/', max_to_keep=1)
    # Restore from the latest checkpoint if requiered
    if ckpt_manager.latest_checkpoint and args.resume:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Last checkpoint restored.")
    # to save the model in tf 2.1.0
    #print('Preparing the model to be saved....')
    #for enc_inputs, targets in dataset.take(1):
    #    dec_inputs = targets[:, :-1]
    #    print (enc_inputs.shape, dec_inputs.shape)
    #    transformer._set_inputs(enc_inputs, dec_inputs, True)

    # Train the model
    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = main_train(dataset, val_dataset, transformer, 
                                                                    args.epochs, 100)

    if args.resume:
        # Finish the wandb job
        wandb.finish()
    # Save the while model
    # Save the entire model to a HDF5 file
    print('Saving the model ....')
    transformer.save_weights(os.path.join(args.sm_model_dir, 'transformer'), overwrite=True, save_format='tf')
    #transformer.save_weights(args.sm_model_dir, overwrite=True, save_format='tf')
    # Save the parameters used to construct the model
    print("Saving the model parameters")
    model_info_path = os.path.join(args.output_data_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'vocab_size_enc': num_words_inputs,
            'vocab_size_dec': num_words_output,
            'sos_token_input': sos_token_input,
            'eos_token_input': eos_token_input,
            'sos_token_output': sos_token_output,
            'eos_token_output': eos_token_output,
            'n_layers': args.n_layers,
            'd_model': args.d_model,
            'ffn_dim': args.ffn_dim,
            'n_heads': args.n_heads,
            'drop_rate': args.dropout_rate
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
