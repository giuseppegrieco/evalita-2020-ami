from utils.grid_search import GridSearchCrossValidation
from utils.cross_validation import KFoldCrossValidation
from utils.data_preparation import build_dictionary, encode_with_position_in_dictionary

import pandas as pd

import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

from utils.embeddings import encode_for_w2v, get_w2v

import sys

if(len(sys.argv) < 2):
  print("Usage: python " + sys.argv[0] + " path/to/grid_file")

import json

with open(sys.argv[1]) as json_file:
  grid_specs = json.load(json_file)

from utils.embeddings import get_w2v, encode_for_w2v

from utils.grid_search import GridSearchCrossValidation
from utils.cross_validation import KFoldCrossValidation

from utils.embeddings import get_w2v, encode_for_w2v

import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorboard.plugins.hparams import api as hp
from transformers import AutoTokenizer

import tensorflow as tf

X_train = []

for input_path in grid_specs["input_path"]:
  DS_X = np.load(input_path["input_path"], allow_pickle=True)
  if input_path['word_embedding'] == 'w2v':
    w2v_model = get_w2v(grid_specs['w2v_path'], DS_X)
    DS_X = encode_for_w2v(DS_X)
    if input_path['sentence_padding']:
      DS_X = pad_sequences(DS_X, maxlen=50, padding="post", truncating="post", value=w2v_model.vocab["</s>"].index)
  elif input_path['word_embedding'] == 'elmo' and input_path['sentence_padding']:
    DS_X = pad_sequences(DS_X, maxlen=50, padding="post", truncating="post", value=0)
  if "context_input" in input_path and input_path["context_input"]: 
    DS_X_context = []
    for sentence_arr in DS_X:
      DS_X_context.append(np.hstack(sentence_arr))
    DS_X = np.array(DS_X_context)
  print(DS_X.shape)
  X_train.append(DS_X)

Y_train = []
for target_path in grid_specs['target_path']:
  Y_train.append(np.load(target_path, allow_pickle=True))

cv = KFoldCrossValidation(5, 12345, True, batch_size="full-batch")

model_file = __import__(grid_specs["model_file"])

gs = GridSearchCrossValidation(
  model_file.build_model, 
  cv,
  verbose=True
)

hps = []
for key in grid_specs["hparams"]:
  hps.append(hp.HParam(key, hp.Discrete(grid_specs["hparams"][key])))
gs.run(
  X_train,
  Y_train,
  hps,
  [
    hp.Metric('tr_loss', display_name='Training Loss'),
    hp.Metric('val_loss', display_name='Validation Loss'),
    hp.Metric('tr_f1_score', display_name='Training Score'),
    hp.Metric('val_f1_score', display_name='Validation Score')
  ],
  base_dir = grid_specs["output_dir"],
  skip = grid_specs["skip"]
)