import pandas as pd

from sklearn.manifold import TSNE

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def w2v_plot(model, keys, filename=False):
    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in model.most_similar(word, topn=30):
            words.append(similar_word)
            embeddings.append(model[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    plt.figure(figsize=(16, 9))
    labels = keys
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=0.7, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title("Word2Vec")
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()

from w2v_plot import w2v_plot

from utils.embeddings import get_w2v
from utils.embeddings import model_w2v

import numpy as np


sentences_tokenized = np.load("_dataset/Task_A_input.npy", allow_pickle = True)
words_to_plot = set()
for sentence in sentences_tokenized:
    for word in sentence:
        words_to_plot.add(word)
w2v = get_w2v("_utils_files/word2vec/spacy_m/trained_model-5.model")
w2v_plot(w2v, ["bello", "puttana", "donna", "troia", "uomo", "persona", "bella"])