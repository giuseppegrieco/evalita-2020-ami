from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from utils.metrics import f1_score
from utils.metrics import weighted_binary_crossentropy
from utils.embeddings import get_w2v

import tensorflow.keras.backend as K
import utils.metrics as metric

import numpy as np

def build_model(hparams):
  if hparams['word_embedding'] == 'w2v':
      input_layer = Input(shape=(hparams['max_sequence_length'], ))
      embedding_layer = get_w2v('').get_keras_embedding(train_embeddings=hparams['train_embeddings'])(input_layer)
  if hparams['word_embedding'] == 'elmo':
      input_layer = Input(shape=(hparams['max_sequence_length'], 1024, ))
      embedding_layer = input_layer
  if hparams["word_embedding"] == "random":
    input_layer = Input(shape=(hparams['max_sequence_length'], ))
    embedding_layer = Embedding(
      hparams["dictionary_len"] + 2,
      hparams["embedding_size"],
      input_length = hparams["max_sequence_length"],
      embeddings_initializer = initializers.RandomNormal(
        mean=0., 
        stddev = 2 / hparams["max_sequence_length"]
      )
    )(input_layer)
  flatten_layer = Flatten()(embedding_layer)
  dropout_layer_1 = Dropout(hparams["dropout_ratio"])(flatten_layer)
  hidden_layer = Dense(
    hparams["hidden_size"], 
    activation = 'relu', 
    kernel_initializer = initializers.RandomUniform(
      minval = - 1 / np.sqrt(hparams["embedding_size"] * hparams["max_sequence_length"]),
      maxval = 1 / np.sqrt(hparams["embedding_size"] * hparams["max_sequence_length"])
    ),
    bias_initializer = initializers.Zeros(),
    kernel_regularizer = regularizers.l2(hparams['l2_regularization'])
  )(dropout_layer_1)
  dropout_layer_2 = Dropout(hparams["dropout_ratio"])(hidden_layer)
  output_layer = Dense(
    2,
    activation = 'sigmoid',
    kernel_initializer = initializers.RandomUniform(
      minval = - 1 / np.sqrt(hparams["hidden_size"]),
      maxval = 1 / np.sqrt(hparams["hidden_size"])
    ),
    bias_initializer = initializers.Zeros(),
    kernel_regularizer = regularizers.l2(hparams['l2_regularization'])
  )(dropout_layer_2)
  model = Model(inputs=[input_layer], outputs=[output_layer])
  model.compile(
    loss = metric.dice_loss,
    optimizer = Adam(learning_rate = hparams["learning_rate"]),
    metrics = [f1_score]
  )
  return model
