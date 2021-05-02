# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import dill as pickle 
from io import StringIO
import sys
import signal
import traceback

import flask

import pandas as pd
import tensorflow as tf
from model import transformer, create_padding_mask, create_look_ahead_mask 


prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
model_name = 'transformer'

START_TOKEN=[8127]
END_TOKEN=[8128]
SUMM_MAX_LENGTH=30



class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
                cls.model = tf.keras.models.load_model(model_path)
                print('Modelo cargado')
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        print('Input: ', input)
        output = tf.expand_dims(START_TOKEN, 0)
        print('Iniciando proceso')
        for i in range(SUMM_MAX_LENGTH):
            predictions = clf.model(inputs=[input, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, END_TOKEN[0]):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        output = tf.squeeze(output, axis=0)

        return output

    

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None 
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        print('csv a procesar')
        data = flask.request.data.decode('utf-8')
        print('Extraido data csv')
        s = StringIO(data)
        data = pd.read_csv(s, header=None)
        print(data.head())
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = []
    print('Iniciando prediccion')
    for input in data.values:
        predictions.append(ScoringService.predict(input))

    # Convert from numpy back to CSV
    out = StringIO()
    pd.DataFrame({'results':predictions}, index =[0]).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')