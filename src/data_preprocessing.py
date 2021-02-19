import pandas

import numpy as np

import json

from utils.data_preprocessing import profile_tag_processing
from utils.data_preprocessing import remove_link
from utils.data_preprocessing import remove_special_characters
from utils.data_preprocessing import remove_letters_repetition
from utils.data_preprocessing import tweet_tokenize
from utils.data_preprocessing import correct

# Read the TRAINING SET provided for the competition (TASK A)
df = pandas.read_csv(
    '../AMI2020_TestSet/AMI2020_test_raw.tsv', 
    header=0, 
    index_col=0, 
    sep='\t', 
    encoding='UTF-8'
)
#1665:3331
suffix="_500_1000"
df = df[500:-1]

# Tokenize RAW data
DS_X = tweet_tokenize(df)
np.save("_testset/Task_A_input_raw", DS_X)

# Preprocessing
df = remove_link(df)
df = remove_letters_repetition(df)
df = remove_special_characters(df)
df = profile_tag_processing(df)

# Tokenize processed data
DS_X = tweet_tokenize(df, shuffle=False)

corrected_words = {}
with open('_utils_files/abbreviation_dictionary.json', 'r', encoding="utf8") as fp:
    abbreviation_dictionary = json.load(fp)
with open('_utils_files/word_frequency_dictionary.json', 'r', encoding="utf8") as fp:
    word_frequency_dictionary = json.load(fp)

# Complete word with char separated by the space (es. A V A N T I)
completed_DS_X = []
for sentence in DS_X:
    completed_sentence = []
    sentence_len = len(sentence)
    completed_word = ""
    for i in range(0, sentence_len):
        word = sentence[i]
        if len(word) == 1:
            completed_word = completed_word + word
        if not len(word) == 1 or i == sentence_len - 1:
            if (len(completed_word) > 0):
                if completed_word in word_frequency_dictionary:
                    completed_sentence.append(completed_word)
                else:
                    for ch in completed_word:
                        completed_sentence.append(ch)
                completed_word = ""
            if not len(word) == 1:
                completed_sentence.append(word)
    completed_DS_X.append(completed_sentence)

# Abbreviation and correction
corrected_DS_X = []
for sentence in completed_DS_X:
    corrected_sentence = []
    for word in sentence:
        if word in abbreviation_dictionary:
            for abbr in abbreviation_dictionary[word].split(" "):
                corrected_sentence.append(abbr)
        else:
            if word in word_frequency_dictionary:
                corrected_words[word] = word
            else:
                if not word in corrected_words:
                    corrected_words[word] = correct(word)
            corrected_sentence.append(corrected_words[word])
    corrected_DS_X.append(corrected_sentence)

# Appare avanti e indietro
fbc_DS_X = []
for sentence in corrected_DS_X:
    completed_sentence = []
    sentence_len = len(sentence)
    for i in range(0, sentence_len):
        word = sentence[i]
        if not word in word_frequency_dictionary:
            new_word = word
            if i > 0:
                previous_word = sentence[i - 1]
                candidate_new_word = previous_word + word
                if candidate_new_word in word_frequency_dictionary:
                    completed_sentence.pop(-1)
                    completed_sentence.append(candidate_new_word)  
                    continue
            if i < sentence_len - 1:
                next_word = sentence[i + 1]
                candidate_new_word = word + next_word
                if candidate_new_word in word_frequency_dictionary:
                    completed_sentence.append(candidate_new_word) 
                    i += 1
                    continue
        completed_sentence.append(word)
    fbc_DS_X.append(completed_sentence)

DS_X = np.array(fbc_DS_X, dtype=object)

np.save("_testset/Task_A_input" + suffix, DS_X)
#np.save("_testset/Task_A_target" + suffix, DS_Y)