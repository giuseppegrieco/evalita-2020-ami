from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, GlobalMaxPooling1D
from tensorflow.keras.layers import SeparableConv1D, Conv1D
from tensorflow.keras.layers import Dropout, Dense, Concatenate, Dot, Softmax, Add
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers
from keras.utils.vis_utils import plot_model

from utils.metrics import f1_score,dice_loss

from utils.embeddings import get_w2v
from utils.metrics import weighted_binary_crossentropy

import tensorflow as tf

import numpy as np

def build_model(
    hparams
):
    sizes = [
        hparams["model_1_size"],
        hparams["model_2_size"],
        hparams["model_3_size"],
        hparams["model_4_size"]
    ]
    names = [
      "cnn_",
      "hate_",
      "sentiment_",
      "bert_"
    ]
    a = Input(shape=(hparams["model_1_size"], ), name="cnn_input")
    b = Input(shape=(hparams["model_2_size"], ), name="hate_input")
    c = Input(shape=(hparams["model_3_size"], ), name="sentiment_input")
    d = Input(shape=(hparams["model_4_size"], ), name="bert_input")
    e = Input(shape=(hparams["context_size"], ), name="context")

    input_layers = [
        a,
        b,
        c,
        d,
        e
    ]

    context_hidden =  Dense(
            hparams["context_size_hidden"],
            activation = 'tanh', 
            kernel_initializer = initializers.RandomUniform(
                minval = - 1 / np.sqrt(hparams['context_size']),
                maxval = 1 / np.sqrt(hparams['context_size'])
            ),
            bias_initializer = initializers.Zeros(),
            kernel_regularizer = regularizers.l2(hparams['l2_regularization']),
            name="context_hidden_repr"
        )(input_layers[4])
    
    nn_s = []
    hh_s = []
    for i in range(0, 4):
        
        hidden_layer = Dense(
            hparams["context_size_hidden"], 
            activation = 'tanh', 
            kernel_initializer = initializers.RandomUniform(
                minval = - 1 / np.sqrt(sizes[i]),
                maxval = 1 / np.sqrt(sizes[i])
            ),
            bias_initializer = initializers.Zeros(),
            kernel_regularizer = regularizers.l2(hparams['l2_regularization']),
            name=names[i] + "hidden_repr"
        )(input_layers[i])
        hh_s.append(hidden_layer)


        output_layer = Dot(axes=1, name=names[i] + "output")([hidden_layer, context_hidden])
        #output_layer = tf.transpose(output_layer)
        nn_s.append(output_layer)

    softmax_layer = Softmax(name="softmax")(nn_s)
    softmax_layer = tf.reshape(
      softmax_layer,
      [-1, 4, 1]
    )
    stack_layer = tf.stack(hh_s, name="stack_hidden", axis=1)
    
    dot_layer = Dot(axes=(1, 1), name="moe_output")([stack_layer,softmax_layer])
    dot_layer = tf.reshape(dot_layer,[-1,hparams['context_size_hidden']])
    output_layer = Dense(
        2,
        activation = 'sigmoid',
        kernel_initializer = initializers.RandomUniform(
            minval = - 1 / np.sqrt(hparams['context_size_hidden']),
            maxval = 1 / np.sqrt(hparams['context_size_hidden'])
        ),
        bias_initializer = initializers.Zeros(),
        kernel_regularizer = regularizers.l2(hparams['l2_regularization']),
        name="model_output"
    )(dot_layer)
    model = Model(inputs=input_layers, outputs=[output_layer])
    model.compile(
        loss = dice_loss,
        optimizer = Adam(learning_rate = hparams['learning_rate']),
        metrics = [f1_score]
    )
    model.summary()
    plot_model(model, "model_moe.png", show_layer_names=True)
    return model