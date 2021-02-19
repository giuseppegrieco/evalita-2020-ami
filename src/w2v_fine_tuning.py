import gensim
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors

import numpy as np

def w2v_fine_tuning(path_to_model, embedding_size):
    sentences_tokenized = np.load("_dataset/Task_A_input.npy", allow_pickle = True)
    model_base = KeyedVectors.load_word2vec_format(path_to_model + "/model.bin", binary = True)
    model = Word2Vec(size=embedding_size, min_count=1)
    model.build_vocab(sentences_tokenized)
    total_examples = model.corpus_count
    model.build_vocab([list(model_base.vocab.keys())], update=True)
    model.intersect_word2vec_format(path_to_model + "/model.bin", binary = True)
    model.train(sentences_tokenized, total_examples = total_examples, epochs = 5)
    model.save(path_to_model + "/trained_model-5.model")
    model.train(sentences_tokenized, total_examples = total_examples, epochs = 5)
    model.save(path_to_model + "/trained_model-10.model")
    model.train(sentences_tokenized, total_examples = total_examples, epochs = 5)
    model.save(path_to_model + "/trained_model-15.model")
    model.train(sentences_tokenized, total_examples = total_examples, epochs = 5)
    model.save(path_to_model + "/trained_model-20.model")
    model.train(sentences_tokenized, total_examples = total_examples, epochs = 5)
    model.save(path_to_model + "/trained_model-25.model")
    model.train(sentences_tokenized, total_examples = total_examples, epochs = 5)
    model.save(path_to_model + "/trained_model-30.model")

import sys

if(len(sys.argv) < 3)
    print("Usage: python " + sys.argv[0] + " path/to/model/ embedding_size")

path_to_model = sys.argv[1]
embedding_size = int(sys.argv[2])
w2v_fine_tuning(path_to_model, embedding_size)