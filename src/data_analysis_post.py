import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('stopwords')

from statistics import mean, median
import os
import re
import json

from utils.plot_utils import autolabel, enriched_histplot, pie_chart


# Load dataset
DS_X = np.load("_dataset/Task_A_input.npy", allow_pickle=True)
DS_Y = np.load("_dataset/Task_A_target.npy", allow_pickle=True)

# Compute tweets length
tweets_word_length_post = [[], [], []]
for tweet, target in zip(DS_X, DS_Y):
    tweets_word_length_post[sum(target)].append(len(tweet))

# Compute dictionary with frequencies
dictionary = dict()
for (tweet, (misogynous, aggressive)) in zip(DS_X,DS_Y):
    for word in tweet:
        if word not in dictionary: dictionary[word] = [0, 0, 0]
        dictionary[word][misogynous + aggressive] += 1

# Compute word frequency distribution
words_frequency_distribution = np.zeros(11)
for freq in dictionary.values():
    words_frequency_distribution[min(sum(freq), 11) -1] += 1

# Compute most frequent words
ordered_words = [[k,v] for k, v in sorted(dictionary.items(), key=lambda item: sum(item[1]))]
stopwords = nltk.corpus.stopwords.words('italian') + [""]
cleaned_ordered_words = np.array([v for v in ordered_words if v[0] not in stopwords], dtype='object')
words_to_plot = cleaned_ordered_words[-20:]
non_mys_words_dist = np.array([v[0] for v in words_to_plot[:, 1]])
mys_words_dist = np.array([v[1] for v in words_to_plot[:, 1]])
agg_words_dist = np.array([v[2] for v in words_to_plot[:, 1]])


### MAKE THE PLOTS ###
## Some plots utils
class_labels = ["Non-misogynus", "Misogynus", "Aggressive"]
class_colors = ["mediumseagreen", "gold", "orangered"]
x_range = 3*np.arange(3)
plots_folder = "plots/data_analysis_post/"

## Creates plot folder
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
else:
    if not os.path.isdir(plots_folder):
        raise NotADirectoryError(plots_folder)

## Length statistics
fig, axs = plt.subplots(4, 1, sharex='col')
enriched_histplot(axs[0], tweets_word_length_post[0] + tweets_word_length_post[1] + tweets_word_length_post[2],
					ylabel="Global", title="Words per Tweet (Postprocessed)")
enriched_histplot(axs[1], tweets_word_length_post[0], ylabel=class_labels[0])
enriched_histplot(axs[2], tweets_word_length_post[1], ylabel=class_labels[1])
enriched_histplot(axs[3], tweets_word_length_post[2], ylabel=class_labels[2])
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels)
fig.tight_layout()
plt.savefig(plots_folder + "Length_stats.png", format='png')
plt.clf()


## Words frequency distribution
labels = ['1','2','3','4','5','6','7','8','9','10','>10']
b1 = plt.bar(np.arange(11), words_frequency_distribution, tick_label=labels)
plt.gca().set(title="Words Frequency Distribution", ylabel="Words", xlabel="Frequency", axisbelow=True)
autolabel(plt.gca(), b1, words_frequency_distribution/len(dictionary)*100)
plt.grid(True, linewidth=0.5)
plt.tight_layout()
plt.savefig(plots_folder + "Words_frequency_distribution.png", format='png')
plt.clf()

## Most frequent words distribution
plt.barh(np.arange(20), non_mys_words_dist, tick_label=words_to_plot[:, 0], color=class_colors[0], label=class_labels[0])
plt.barh(np.arange(20), mys_words_dist, left=non_mys_words_dist, color=class_colors[1], label=class_labels[1])
plt.barh(np.arange(20), agg_words_dist, left=non_mys_words_dist+mys_words_dist, color=class_colors[2], label=class_labels[2])
plt.gca().set(title="Most frequent words", axisbelow=True)
plt.legend()
plt.grid(True, linewidth=0.5)
plt.tight_layout()
plt.savefig(plots_folder + "Words_Statistics.png", format='png')
plt.clf()


print("Total distinct words: " + str(len(dictionary.keys())))


threshold = 50

count_non_mys = 0
for t in tweets_word_length_post[0]:
	if t > threshold:
		count_non_mys += 1

count_mys = 0
for t in tweets_word_length_post[1]:
	if t > threshold:
		count_mys += 1

count_agg = 0
for t in tweets_word_length_post[1]:
	if t > threshold:
		count_agg += 1

print("Tweets with length > " + str(threshold) + ": ")
print("  total   = " + str(count_non_mys + count_mys + count_agg))
print("  non-mys = " + str(count_non_mys))
print("  mys     = " + str(count_mys))
print("  agg     = " + str(count_agg))
