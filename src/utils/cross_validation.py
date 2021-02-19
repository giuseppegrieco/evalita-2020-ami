import abc
import time
import statistics
import gc

from .data_preparation import encode_classes

from sklearn.model_selection import StratifiedKFold

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from tensorflow.python.keras import backend as K

import inspect

import numpy as np

import math
from keras.utils.data_utils import Sequence

class BalancedBatchGenerator(Sequence):
    def __init__(self, X, Y, batch_size=32):
        self.batch_size = min(batch_size, len(X[0]))
        self.len = math.floor(len(X[0]) / self.batch_size)
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx == 0:
            self._indices = []
            for train_indices, validation_indices in StratifiedKFold(n_splits = self.len, shuffle=True).split(self.X[0], encode_classes(self.Y[0])):
                self._indices.append(validation_indices)
        X_res = []
        Y_res = []
        for xs in self.X:
            X_res.append(xs[self._indices[idx]])
        for ys in self.Y:
            Y_res.append(ys[self._indices[idx]])
        return X_res, Y_res
        
class CrossValidation(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def evaluate(self, model_fn, X, Y, training_parameters):
        pass

class KFoldCrossValidation(CrossValidation):
    def __init__(self, k, random_state = None, verbose = False, **evaluation_parameters):
        self._k = k
        self._random_state = random_state
        self._verbose = verbose
        evaluation_parameters["verbose"] = verbose
        self._evaluation_parameters = evaluation_parameters

    def evaluate(self, model_fn, X, Y, hparams, logdir, run, metric):
        tr_scores = []
        vl_scores = []
        for i, (train_indices, validation_indices) in enumerate(
            StratifiedKFold(n_splits = self._k, random_state = self._random_state, shuffle=True).split(X[0], encode_classes(Y[0]))
        ):
            fold_number = str(i + 1)
            if(self._verbose):
                print("Working on fold n." + fold_number)

            model = model_fn()

            current_evaluation_parameters = self._evaluation_parameters.copy()
            X_tr = []
            X_vl = []
            Y_tr = []
            Y_vl = []
            for Xi in X:
                X_tr.append(Xi[train_indices])
                X_vl.append(Xi[validation_indices])
            for Yi in Y:
                Y_tr.append(Yi[train_indices])
                Y_vl.append(Yi[validation_indices])

            current_training_parameters = {}
            available_training_parameters = set([
                'epochs', 
                'verbose', 
                'callbacks'
                'validation_split',
                'validation_data', 
                'shuffle', 
                'class_weight',
                'sample_weight', 
                'initial_epoch', 
                'steps_per_epoch',
                'validation_steps', 
                'validation_batch_size', 
                'validation_freq',
                'max_queue_size',
                'workers', 
                'use_multiprocessing'
            ])
            for key in hparams:
                if key in available_training_parameters:
                    current_training_parameters[key] = hparams[key]

            current_training_parameters["validation_data"] = (X_vl, Y_vl)
            current_training_parameters["verbose"] = self._verbose

            if "batch_size" in current_training_parameters and current_training_parameters["batch_size"] == "full-batch":
                current_training_parameters["batch_size"] = len(X_tr[0])
            if "validation_batch_size" in current_training_parameters and current_training_parameters["validation_batch_size"] == "full-batch":
                current_training_parameters["validation_batch_size"] = len(X_vl[0])
                
            if "batch_size" in current_evaluation_parameters and current_evaluation_parameters["batch_size"] == "full-batch":
                current_evaluation_parameters["batch_size"] = len(X_vl[0])  

            run_dir = logdir + str(run) + '-fold-' + fold_number + '/'
            with tf.summary.create_file_writer(run_dir).as_default():
                hp.hparams(hparams, trial_id = str(run) + "-fold" + fold_number)
                exec_time = time.time()
                if not 'callbacks' in current_training_parameters:
                    current_training_parameters['callbacks'] = []
                current_training_parameters['callbacks'].append(
                    tf.keras.callbacks.TensorBoard(run_dir)
                )
                current_training_parameters['callbacks'].append(
                    hp.KerasCallback(run_dir, hparams, trial_id = str(run) + "-fold-" + fold_number)
                )
                if hparams['patience']:
                    current_training_parameters['callbacks'].append(
                        tf.keras.callbacks.EarlyStopping(
                            monitor = "val_f1_score", 
                            patience = hparams['patience'], 
                            mode = "max", 
                            restore_best_weights=True
                        )
                    )
                history = model.fit(
                    BalancedBatchGenerator(X_tr, Y_tr, hparams["batch_size"]),
                    **current_training_parameters
                )
                tr_score = model.evaluate(X_tr, Y_tr, **current_evaluation_parameters)
                tr_scores.append(tr_score)  
                vl_score = model.evaluate(X_vl, Y_vl, **current_evaluation_parameters)
                vl_scores.append(vl_score)
                for j, metric_name in enumerate(model.metrics_names):
                    tf.summary.scalar("tr_" + metric_name, tr_score[j], step=1)
                    tf.summary.scalar("val_" + metric_name, vl_score[j], step=1)
            K.clear_session()
            if i < 4:
              del model
            gc.collect()

        run_dir = logdir + str(run) + '-mean/'
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams, trial_id = str(run) + "-mean")
            for j, (metric_name, tr_metric_scores, vl_metric_scores) in enumerate(zip(model.metrics_names, zip(*tr_scores), zip(*vl_scores))):
                tf.summary.scalar("tr_" + metric_name, statistics.mean(tr_metric_scores), step=1)
                tf.summary.scalar("val_" + metric_name, statistics.mean(vl_metric_scores), step=1)
        run_dir = logdir + str(run) + '-stdev/'
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams, trial_id = str(run) + "-stdev")
            for j, (metric_name, tr_metric_scores, vl_metric_scores) in enumerate(zip(model.metrics_names, zip(*tr_scores), zip(*vl_scores))):
                tf.summary.scalar("tr_" + metric_name, statistics.stdev(tr_metric_scores), step=1)
                tf.summary.scalar("val_" + metric_name, statistics.stdev(vl_metric_scores), step=1)
            del model

    def folds(self):
        return []