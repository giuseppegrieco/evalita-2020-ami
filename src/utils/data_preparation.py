import numpy as np

def encode_with_position_in_dictionary(DS_X, dictionary):
    """
        TODO: documentation
    """
    DS_X_encoded = []
    for sentence in DS_X:
        sentence_encoded = []
        for word in sentence:
            if word in dictionary:
                sentence_encoded.append(dictionary[word])
            else:
                sentence_encoded.append(1)
        DS_X_encoded.append(sentence_encoded)
    return DS_X_encoded

def encode_classes(Y):
    res = []
    for v in Y:
        if v[0] == 1 and v[1] == 1:
            res.append(0)
        else:
            if v[0] == 1:
                res.append(1)
            else:
                res.append(2)
    return np.array(res)

def build_dictionary(DS_X):
    """
        TODO: documentation
    """
    dictionary = {}
    i = 2
    for sentence in DS_X:
        for word in sentence:
            if not (word in dictionary):
                dictionary[word] = i
                i += 1
    return dictionary