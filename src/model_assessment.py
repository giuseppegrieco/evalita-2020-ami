import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

from utils.embeddings import encode_for_w2v, get_w2v
from utils.data_preparation import encode_classes


# Load dataset
DS_X = np.load("_dataset/Task_A_input.npy", allow_pickle=True)
DS_Y = np.load("_dataset/Task_A_target.npy", allow_pickle=True)

# Load word2vec model and prepare dataset for training 
w2v_path = '_utils_files/word2vec/nlpl/trained_model-30.model'                                           # <============= INSERT EMBEDDINGS PATH HERE
w2v_model = get_w2v(w2v_path, DS_X, train_purpose=False)
DS_X = encode_for_w2v(DS_X)
DS_X = pad_sequences(DS_X, maxlen=70, padding="post", value=w2v_model.vocab["</s>"].index)

# Dive dataset in training and validation
skf = StratifiedKFold(n_splits = 10, random_state = 79946, shuffle=True)
for train_indices, validation_indices in skf.split(DS_X, encode_classes(DS_Y)):
    X_tr = DS_X[train_indices]
    Y_tr = DS_Y[train_indices]
    X_vl = DS_X[validation_indices]
    Y_vl = DS_Y[validation_indices]
    break

# HYPERPARAMETERS
model_parameters = {
    'dropout_ratio': 0.7,
    'filter_param': 75,
    'hidden_size': 75,
    'kernel_sizes': [3,5,7,9],
    'l2_regularization': 1e-4,
    'learning_rate': 1e-3,
    'max_sequence_length': 70,
    'max_norm_value': 3,
    'w2v_m': w2v_model
}

training_parameters = {
    'epochs': 100,
    'batch_size': 64,
    'shuffle': True,
    'validation_data': (X_vl,Y_vl),
    'validation_batch_size': len(X_vl),
    'callbacks': [tf.keras.callbacks.EarlyStopping(
                    monitor = "val_f1_score", 
                    patience = 10, 
                    mode = "max", 
                    restore_best_weights=True
        )]
}

# Create and train the model
model_file = __import__("model_cnn_multichannel")
model = model_file.build_model(**model_parameters)
model.fit(X_tr, Y_tr, **training_parameters)

vl_score = model.evaluate(X_vl, Y_vl, batch_size=len(X_vl))
print(vl_score)


# Load the preprocessed test set and encode it
Test_X_0_500 = np.load("_testset/Task_A_input_0_500.npy", allow_pickle=True)
Test_X_500_999 = np.load("_testset/Task_A_input_500_1000.npy", allow_pickle=True)
Test_X_999_1000 = np.load("_testset/Task_A_input_999_1000.npy", allow_pickle=True)
p = np.empty((1,), dtype=object)
p[0] = Test_X_999_1000.tolist()
Test_X = np.block([Test_X_0_500, Test_X_500_999, p])
print("Test set shape: " + str(Test_X.shape))

Test_X = encode_for_w2v(Test_X)
Test_X = pad_sequences(Test_X, maxlen=70, padding="post", value=w2v_model.vocab["</s>"].index)

# Predict and save predictions
results = model.predict(Test_X)
results = np.rint(results)
np.savetxt("results.tsv", results, fmt="%d", delimiter='\t')
