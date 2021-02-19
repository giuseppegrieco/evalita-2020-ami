import abc
import time
import statistics 
import datetime

import pandas as pd

from itertools import product

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

class GridSearchParameters():
    def __init__(self, hparams):
        self.__hparams = hparams

    def __iter__(self):
        sorted_items = sorted(self.__hparams, key = lambda x: x.name)
        if not sorted_items:
            yield {}
        else:
            keys = []
            values = []
            for hparam in sorted_items:
                keys.append(hparam.name)
                values.append(hparam.domain.values)
            for value in product(*values):
                parameters = dict(zip(keys, value))
                yield parameters

class GridSearchBase(metaclass=abc.ABCMeta):
    def __init__(self, model_function, verbose = False):
        self._model_funtion = model_function
        self._verbose = verbose

    def run(self, 
            X, 
            Y, 
            hparams,
            metrics,
            base_dir = '',
            skip = -1
    ):
        log_dir = base_dir + 'logs/hparam_tuning-' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'
        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams_config(
                hparams = hparams,
                metrics = metrics
            )
        i = 0
        for hparams in GridSearchParameters(hparams):
            if i <= skip:
                i += 1
                continue
            if(self._verbose):
                print("Hyperparameters combination n." + str(i + 1))
                print(hparams)
            
            self.evaluate(X, Y, hparams, log_dir, i, metrics)
            i += 1
    
    @abc.abstractclassmethod
    def evaluate(self, X, Y, hparams, logidr, run):
        pass

class GridSearchCrossValidation(GridSearchBase):
    def __init__(self, model_function, cross_validation, verbose = False):
        super().__init__(model_function, verbose)
        self._cross_validation = cross_validation
    
    def evaluate(self, X, Y, hparams, log_dir, run, metrics):
        model_fn = lambda: self._model_funtion(hparams)
        result = self._cross_validation.evaluate(model_fn, X, Y, hparams, log_dir, run, metrics)
        return result