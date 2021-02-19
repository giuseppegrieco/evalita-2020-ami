#from elmoformanylangs import Embedder
from gensim.models import KeyedVectors, Word2Vec

from .data_preparation import encode_with_position_in_dictionary    
from .data_preparation import build_dictionary

model_w2v = None
model_elmo = None

def get_w2v(model_file, DS_X = None, train_purpose = True):
    global model_w2v
    if model_w2v == None:
        model = Word2Vec.load(model_file)
        model_w2v = model.wv
        if train_purpose:
            ds_dict = build_dictionary(DS_X)
            #ds_dict["</s>"] = -5
            new_model_w2v = KeyedVectors(len(model_w2v["</s>"]))
            for word in ds_dict:
                new_model_w2v.add([word], [model_w2v[word]])
            new_model_w2v.add(["</s>"], [0*model_w2v["ciao"]])
            print(new_model_w2v["</s>"])
            model_w2v = new_model_w2v
    return model_w2v

def encode_for_w2v(DS_X):
    DS_X_encoded = []
    for sentence in DS_X:
        sentence_encoded = []
        for word in sentence:
            if word in model_w2v.vocab:
                sentence_encoded.append(model_w2v.vocab[word].index)
        DS_X_encoded.append(sentence_encoded)
    return DS_X_encoded

def get_elmo():
    global model_elmo
    if model_elmo == None:
        model_elmo = Embedder(model_dir='_utils_files/it.model/')
    return model_elmo

def encode_with_elmo(DS_X):
    _model_elmo = get_elmo()
    DS_X_encoded = _model_elmo.sents2elmo(DS_X, output_layer=2)
    return DS_X_encoded