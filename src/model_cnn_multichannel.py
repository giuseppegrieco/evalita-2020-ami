from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, GlobalMaxPooling1D
from tensorflow.keras.layers import SeparableConv1D, Conv1D
from tensorflow.keras.layers import Dropout, Dense, Concatenate
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers

from utils.metrics import f1_score,dice_loss

from utils.embeddings import get_w2v
from utils.metrics import weighted_binary_crossentropy

import numpy as np

def build_model(
    hparams
):
    input_layer = Input(shape=(hparams["max_sequence_length"], ))
    embedding_layer_static = get_w2v('').get_keras_embedding(train_embeddings=False)(input_layer)
    embedding_layer = get_w2v('').get_keras_embedding(train_embeddings=True)(input_layer)
    
    submodels = []
    kernel_sizes = hparams['kernel_sizes'].split('-')
    for ks in kernel_sizes:
        model = Sequential()
        conv_1_d = Conv1D(
            activation = 'relu',
            filters = hparams["filters"], 
            kernel_size = int(ks),
            kernel_constraint = max_norm(hparams["max_norm_value"])
        )
        conv_layer_static = conv_1_d(embedding_layer_static)
        conv_layer = conv_1_d(embedding_layer)
        max_pooling_static = GlobalMaxPooling1D()(conv_layer_static)
        max_pooling = GlobalMaxPooling1D()(conv_layer)
        concatenate_layer = Concatenate()([max_pooling_static, max_pooling])
        submodels.append(concatenate_layer)
    concat = Concatenate()(submodels)
    dropout_layer_1 = Dropout(hparams['dropout_ratio'])(concat)
    hidden_layer = Dense(
        hparams['hidden_size'], 
        activation = 'relu', 
        kernel_initializer = initializers.RandomUniform(
            minval = - 1 / np.sqrt(len(kernel_sizes) * 2* hparams['filters']),
            maxval = 1 / np.sqrt(len(kernel_sizes) * 2 * hparams['filters'])
        ),
        bias_initializer = initializers.Zeros(),
        kernel_regularizer = regularizers.l2(hparams['l2_regularization'])
    )(dropout_layer_1)
    dropout_layer_2 = Dropout(hparams['dropout_ratio'])(hidden_layer)
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
        loss = dice_loss,
        optimizer = Adam(learning_rate = hparams["learning_rate"]),
        metrics = [f1_score]
    )
    from keras.utils.vis_utils import plot_model
    plot_model(model, "model_cnn_multichannel.png", show_layer_names=False)
    return model