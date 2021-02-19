'''
    1. prendiamo tutti gli output per ogni loss
    2. per ogni loss:
        1. confusion matrix
        2. AUC
        3. Grafico loss - f1
'''

import numpy as np
from matplotlib import pyplot as plt

import tensorflow.keras.backend as K
import utils.metrics as metric
import pandas

names=["Dice","Dice + WBC", "Differentiable F1", "Weighted Binary Cross-entropy"]

for j in range(0, 4):
    tr_f1_score_path = "_utils_files/loss_eval/run-logs_hparam_tuning-2020-10-13_16-25-31_"+str(j)+"-fold-4_train-tag-epoch_f1_score.csv"
    tr_loss_path = "_utils_files/loss_eval/run-logs_hparam_tuning-2020-10-13_16-25-31_"+str(j)+"-fold-4_train-tag-epoch_loss.csv"
    vl_f1_score_path = "_utils_files/loss_eval/run-logs_hparam_tuning-2020-10-13_16-25-31_"+str(j)+"-fold-4_validation-tag-epoch_f1_score.csv"
    vl_loss_path = "_utils_files/loss_eval/run-logs_hparam_tuning-2020-10-13_16-25-31_"+str(j)+"-fold-4_validation-tag-epoch_loss.csv"

    tr_f1_score = pandas.read_csv(tr_f1_score_path)
    epochs = tr_f1_score.Step.values[-1] + 1
    tr_f1_score = tr_f1_score.Value.values
    tr_loss = pandas.read_csv(tr_loss_path).Value.values

    vl_f1_score = pandas.read_csv(vl_f1_score_path).Value.values
    vl_loss = pandas.read_csv(vl_loss_path).Value.values

    _TR_F1 = 1 -  tr_f1_score[0]
    _TR_LOSS = tr_loss[0]
    tr_diff_f1 = []
    tr_diff_loss = []

    _VL_F1 = 1 -  vl_f1_score[0]
    _VL_LOSS = vl_loss[0]
    vl_diff_f1 = []
    vl_diff_loss = []

    for i in range(1, epochs):
        _TR_F1_DIFF = ((1 - tr_f1_score[i]) - _TR_F1) / _TR_F1
        _TR_F1 = 1 - tr_f1_score[i]

        _TR_LOSS_DIFF = (tr_loss[i] - _TR_LOSS) / _TR_LOSS
        _TR_LOSS = tr_loss[i]

        tr_diff_f1.append(_TR_F1_DIFF)
        tr_diff_loss.append(_TR_LOSS_DIFF)

        _VL_F1_DIFF = ((1 - vl_f1_score[i]) - _VL_F1) / _VL_F1
        _VL_F1 = 1 - vl_f1_score[i]

        _VL_LOSS_DIFF = (vl_loss[i] - _VL_LOSS) / _VL_LOSS
        _VL_LOSS = vl_loss[i]

        vl_diff_f1.append(_VL_F1_DIFF)
        vl_diff_loss.append(_VL_LOSS_DIFF)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1)
    axs[0].scatter(np.arange(epochs - 1), tr_diff_f1, label="f1 score")
    axs[0].plot([0, epochs], [0, 0], color="black")
    axs[0].scatter(np.arange(epochs - 1), tr_diff_loss, label="loss")
    axs[0].set_title("Training")
    axs[0].legend()
    axs[1].scatter(np.arange(epochs - 1), vl_diff_f1, label="f1 score")
    axs[1].plot([0, epochs], [0, 0], color="black")
    axs[1].scatter(np.arange(epochs - 1), vl_diff_loss, label="loss")
    axs[1].set_title("Validation")
    axs[1].legend()
    for ax in axs:
        ax.set_xlabel("epochs")
        ax.set_ylabel("growth ratio")
        ax.set_ylim(-0.5, 0.5)
    fig.tight_layout()
    fig.suptitle(names[j])
    fig.savefig(names[j] + ".png")
    fig.savefig(names[j] + ".svg")
    plt.show()