import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, GlobalMaxPooling1D
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

def build_submodels(kernel_sizes,filter_param,max_norm_value,input_layer):
    submodels = []
    kernel_sizes = kernel_sizes.split('-')
    for ks in kernel_sizes:
        conv_1_d = Conv1D(
            activation = 'relu',
            filters = filter_param, 
            kernel_size = int(ks),
            kernel_constraint = max_norm(max_norm_value)
        )
        conv_layer = conv_1_d(input_layer)
        max_pooling = GlobalMaxPooling1D()(conv_layer)
        submodels.append(max_pooling)
    return submodels

def build_model(hparams): 

    input_layer_dynamic = Input(shape=(hparams['max_sequence_length'],), name='w2v_input')
    input_layer_static = Input(shape=(hparams['max_sequence_length'],hparams['embedding_size']),name='ELMo_input')

    embedding_layer = get_w2v('').get_keras_embedding(train_embeddings=True)(input_layer_dynamic)
    
    submodels = []
    submodels.extend(build_submodels(hparams['kernel_sizes'],hparams['filters'],
                    hparams['max_norm_value'],embedding_layer))
    submodels.extend(build_submodels(hparams['kernel_sizes'],hparams['filters'],
                    hparams['max_norm_value'],input_layer_static))
    
    concat = Concatenate()(submodels)

    dropout_layer_1 = Dropout(hparams['dropout_ratio'])(concat)
    hidden_layer = Dense(
        hparams['hidden_size'], 
        activation = 'relu', 
        kernel_initializer = initializers.RandomUniform(
            minval = - 1 / np.sqrt(2 * len(hparams['kernel_sizes'])*hparams['filters']),
            maxval = 1 / np.sqrt(2 * len(hparams['kernel_sizes'])*hparams['filters'] )
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
    
    model = Model(inputs=[input_layer_dynamic,input_layer_static], outputs=output_layer)
    model.compile(
        loss = metric.dice_loss,
        optimizer = Adam(learning_rate = hparams['learning_rate']),
        metrics = [f1_score]
    )
    #model.summary()

    return model