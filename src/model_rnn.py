from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.layers import Dropout, Dense, Bidirectional, GRU
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers

from utils.metrics import f1_score
from utils.metrics import weighted_binary_crossentropy
from utils.embeddings import get_w2v

import tensorflow.keras.backend as K
import utils.metrics as metric

import tensorflow as tf

import numpy as np

def build_model(hparams):
    hidden_sizes = hparams['hidden_sizes_rnn'].split('-')
    input_layer = Input(shape=(hparams['max_sequence_length'], ))
    if hparams["word_embedding"] == "w2v":
        embedding_layer = get_w2v('').get_keras_embedding(train_embeddings=hparams['train_embeddings'])(input_layer)
    if hparams["word_embedding"] == "random":
        embedding_layer = Embedding(
            hparams["dictionary_len"] + 2,
            hparams["embedding_size"],
            input_length = hparams["max_sequence_length"],
            embeddings_initializer = initializers.RandomNormal(
                mean=0., 
                stddev = 2 / hparams["max_sequence_length"]
            )
        )(input_layer)
    bidirection_layer_1 = Bidirectional(GRU(int(hidden_sizes[0]), activation='relu', return_sequences=True))(embedding_layer)
    bidirection_layer_2 = Bidirectional(GRU(int(hidden_sizes[1]), activation='relu', return_sequences=True))(bidirection_layer_1)
    bidirection_layer_3 = Bidirectional(GRU(int(hidden_sizes[2]), activation='relu'), merge_mode="concat")(bidirection_layer_2)
    dropout_layer_1 = Dropout(hparams["dropout_ratio"])(bidirection_layer_3)
    output_layer = Dense(
        2,
        activation = 'sigmoid',
        kernel_initializer = initializers.RandomUniform(
            minval = - 1 / np.sqrt(int(hidden_sizes[2])),
            maxval = 1 / np.sqrt(int(hidden_sizes[2]))
        ),
        bias_initializer = initializers.Zeros(),
        kernel_regularizer = regularizers.l2(hparams['l2_regularization'])
    )(dropout_layer_1)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(
        loss = metric.dice_loss,
        optimizer = RMSprop(learning_rate = hparams["learning_rate"],momentum=0.9),
        metrics = [f1_score]
    )
    tf.keras.utils.plot_model(model, "model_rnn.png")

    model.summary()
    return model