import numpy as np
from tensorflow.keras.layers import Input, Flatten,Lambda
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, regularizers, initializers
from utils.metrics import f1_score,dice_loss
from transformers import AutoTokenizer, TFAutoModel
from keras.utils.vis_utils import plot_model
#"m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"
import tensorflow as tf

def build_model(
    hparams):

    bert_model = TFAutoModel.from_pretrained(hparams["bert_file_name"])
    bert_model.trainable = True

    if  not hparams['trainable_bert'] is None:
        bert_model.trainable = hparams['trainable_bert']


    input_layer_ids = Input(shape = (hparams['max_sequence_length'],), dtype='int64')
    input_layer_masks = Input(shape = (hparams['max_sequence_length'],), dtype='int64')
    bert_output = bert_model([input_layer_ids,input_layer_masks])
    bert_output = bert_output[1]

    classifier = Dense(units = 2,
        activation = 'sigmoid',
        kernel_initializer = initializers.RandomUniform(
            minval = - 1 / np.sqrt(bert_output.shape[1]),
            maxval = 1 / np.sqrt(bert_output.shape[1])
        ),
        bias_initializer = initializers.Zeros(),
        kernel_regularizer = regularizers.l2(hparams['l2_regularization'])
    )(bert_output)

    model = Model(inputs=[input_layer_ids,input_layer_masks], outputs=classifier)
    model.compile(
        loss= dice_loss,
        optimizer = Adam(learning_rate = hparams["learning_rate"]),
        metrics = [f1_score]
    )
    plot_model(model, "model_bert.png", show_layer_names=False)
    return model


def save_pooler_output(bert_model, ids, mask, step, path):
    res = []

    for i in range(0, len(ids), step):

        temp_ids = ids[i:i + step]
        temp_mask = mask[i:i+step]

        out = bert_model([temp_ids, temp_mask])[1]
        print(out.shape)
        res.append(out)

    res = np.concatenate(res, axis=0)
    print(res.shape)

    np.save(path, res)