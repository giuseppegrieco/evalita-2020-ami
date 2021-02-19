from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout, Dense, Concatenate
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers

from utils.metrics import f1_score
from utils.embeddings import get_w2v

import utils.metrics as metric

import numpy as np

def build_model(
    hparams
):
    if hparams['word_embedding'] == 'w2v':
        input_layer = Input(shape=(hparams['max_sequence_length'], ))
        embedding_layer = get_w2v('').get_keras_embedding(train_embeddings=hparams['train_embeddings'])(input_layer)
    if hparams['word_embedding'] == 'elmo':
        input_layer = Input(shape=(hparams['max_sequence_length'], 1024, ))
        embedding_layer = input_layer

    submodels = []
    kernel_sizes = hparams['kernel_sizes'].split('-')
    for ks in kernel_sizes:
        model = Sequential()
        conv_layer = Conv1D(
            activation = 'relu',
            filters = hparams['filters'], 
            kernel_size = int(ks),
            kernel_constraint = max_norm(hparams['max_norm_value'])
        )(embedding_layer)
        max_pooling = GlobalMaxPooling1D()(conv_layer)
        submodels.append(max_pooling)
    concat = Concatenate()(submodels)

    dropout_layer_1 = Dropout(hparams['dropout_ratio'])(concat)
    hidden_layer = Dense(
        hparams['hidden_size'], 
        activation = 'relu', 
        kernel_initializer = initializers.RandomUniform(
            minval = - 1 / np.sqrt(len(kernel_sizes) * hparams['filters']),
            maxval = 1 / np.sqrt(len(kernel_sizes) * hparams['filters'])
        ),
        bias_initializer = initializers.Zeros(),
        kernel_regularizer = regularizers.l2(hparams['l2_regularization'])
    )(dropout_layer_1)
    dropout_layer_2 = Dropout(hparams['dropout_ratio'])(concat)
    output_layer = Dense(
        2,
        activation = 'sigmoid',
        kernel_initializer = initializers.RandomUniform(
            minval = - 1 / np.sqrt(hparams['hidden_size']),
            maxval = 1 / np.sqrt(hparams['hidden_size'])
        ),
        bias_initializer = initializers.Zeros(),
        kernel_regularizer = regularizers.l2(hparams['l2_regularization'])
    )(dropout_layer_2)
    
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(
        loss = metric.dice_loss,
        optimizer = Adam(learning_rate = hparams['learning_rate']),
        metrics = [f1_score]
    )

    return model