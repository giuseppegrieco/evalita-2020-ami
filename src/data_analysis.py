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


# Load the dataset
df = pd.read_csv('../AMI2020_TrainingSet/AMI2020_training_raw.tsv', header=0, index_col=0, sep='\t', encoding='UTF-8')

# Data in Numpy array
data_np = df.values

# Data distribution
df_non_misogynous = df.loc[df.misogynous == 0]
df_misogynous = df.loc[df.misogynous == 1]
df_misogynous_non_aggressive = df_misogynous.loc[df.aggressiveness == 0]
df_misogynous_aggressive = df_misogynous.loc[df.aggressiveness == 1]
class_counts = np.array([len(df_non_misogynous), len(df_misogynous_non_aggressive), len(df_misogynous_aggressive)])

# Counts hashtags distribution
hashtag_counts = np.array([0, 0, 0])
for (tweet, misogynous, aggressive) in data_np:
    for word in tweet.split(" "):
        if '#' in word:
            hashtag_counts[misogynous + aggressive] += 1
            break
hashtag_percentages = hashtag_counts / class_counts * 100

# Counts metions distribution
mention_counts = np.array([0, 0, 0])
for (tweet, misogynous, aggressive) in data_np:
    for word in tweet.split(" "):
        if '@' in word:
            mention_counts[misogynous + aggressive] += 1
            break
mention_percentages = mention_counts / class_counts * 100

# Counts links distribution
link_counts = np.array([0, 0, 0])
for (tweet, misogynous, aggressive) in data_np:
    for word in tweet.split(" "):
        if 'http' in word:
            link_counts[misogynous + aggressive] += 1
            break
link_percentages = link_counts / class_counts * 100

# Compute tweet lengths in characters and words
tweet_char_lengths = [[], [], []]
tweet_word_lengths = [[], [], []]
for (tweet, misogynous, aggressive) in data_np:
    tweet_char_lengths[misogynous + aggressive].append(len(tweet))
    tweet_word_lengths[misogynous + aggressive].append(len(tweet.split(" ")))

tweet_char_length = tweet_char_lengths[0] + tweet_char_lengths[1] + tweet_char_lengths[2]
tweet_word_length = tweet_word_lengths[0] + tweet_word_lengths[1] + tweet_word_lengths[2]

# Gets most frequent words distribution
dictionary = dict()
for (tweet, misogynous, aggressive) in data_np:
    for word in tweet.lower().split(" "):
        if not (('http' in word) or ('@' in word)):
            word = re.sub("[^\w\s]", "", word)
            if word not in dictionary: dictionary[word] = [0, 0, 0]
            dictionary[word][misogynous + aggressive] += 1

ordered_words = [[k,v] for k, v in sorted(dictionary.items(), key=lambda item: sum(item[1]))]
stopwords = nltk.corpus.stopwords.words('italian') + [""]
cleaned_ordered_words = np.array([v for v in ordered_words if v[0] not in stopwords], dtype='object')
words_to_plot = cleaned_ordered_words[-20:]

non_mys_words_dist = np.array([v[0] for v in words_to_plot[:, 1]])
mys_words_dist = np.array([v[1] for v in words_to_plot[:, 1]])
agg_words_dist = np.array([v[2] for v in words_to_plot[:, 1]])

# Compute words frequency distribution
words_frequency_distribution = np.zeros(11)
for w, freqs in ordered_words:
    words_frequency_distribution[min(sum(freqs), 11) -1] += 1

# Gets out of vocabulary words distribution
OOV_words = dict()
with open('_utils_files/word_frequency_dictionary.json', 'r', encoding="utf8") as fp:
    preprocessing_dictionary = json.load(fp)
    
for word in dictionary.keys():
    if not word in preprocessing_dictionary:
        OOV_words[word] = 1

OOV_words_distribution = [len(dictionary) - len(OOV_words), len(OOV_words)]

OOV_words_frequency = [0, 0]
for w, freqs in ordered_words:
    OOV_words_frequency[w in OOV_words] += sum(freqs)



### MAKE THE PLOTS ###
## Some plots utils
class_labels = ["Non-misogynus", "Misogynus", "Aggressive"]
class_colors = ["mediumseagreen", "gold", "orangered"]
x_range = 3*np.arange(3)
plots_folder = "plots/data_analysis/"

## Creates plot folder
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
else:
    if not os.path.isdir(plots_folder):
        raise NotADirectoryError(plots_folder)

## Classes distribution plot
pie_chart(plt.gca(), class_counts, labels=class_labels, colors=class_colors, title="Classes Distribution")
plt.tight_layout()
plt.savefig(plots_folder + "Classes_distribution.png", format='png')
plt.clf()

## Text statistics per class plot
b1 = plt.bar(x_range-0.9, class_counts,   width=0.6, label='Total tweets',         color='cornflowerblue')
b2 = plt.bar(x_range-0.3, mention_counts, width=0.6, label='Tweets with mentions', color='coral')
b3 = plt.bar(x_range+0.3, hashtag_counts, width=0.6, label='Tweets with hashtags', color='palegreen')
b4 = plt.bar(x_range+0.9, link_counts,    width=0.6, label='Tweets with links',    color='plum')
autolabel(plt.gca(), b2, mention_percentages)
autolabel(plt.gca(), b3, hashtag_percentages)
autolabel(plt.gca(), b4, link_percentages)
plt.gca().set(xticks=x_range, xticklabels=class_labels, ylabel="Tweets", axisbelow=True, title="Per class text statistics")
plt.legend(loc='upper center')
plt.grid(True, linewidth=0.5)
plt.tight_layout()
plt.savefig(plots_folder + "Statistics.png", format='png')
plt.clf()

## Global length statistics
fig, [ax1, ax2] = plt.subplots(1, 2)
enriched_histplot(ax1, tweet_char_length, ylabel="Tweets", title="Chars per Tweet", legend=True)
enriched_histplot(ax2, tweet_word_length, ylabel="Tweets", title="Words per Tweet", legend=True)
plt.tight_layout()
plt.savefig(plots_folder + "Lenghts_global.png", format='png')
plt.clf()

## Per class length statistics
fig, [[ax11, ax12], [ax21, ax22], [ax31, ax32]] = plt.subplots(3, 2, sharex='col', sharey='row')
enriched_histplot(ax11, tweet_char_lengths[0], title="Chars per Tweet", ylabel=class_labels[0])
enriched_histplot(ax12, tweet_word_lengths[0], title="Words per Tweet")
enriched_histplot(ax21, tweet_char_lengths[1], ylabel=class_labels[1])
enriched_histplot(ax22, tweet_word_lengths[1])
enriched_histplot(ax31, tweet_char_lengths[2], ylabel=class_labels[2])
enriched_histplot(ax32, tweet_word_lengths[2])
handles, labels = ax11.get_legend_handles_labels()
fig.legend(handles, labels, loc='right')
fig.tight_layout()
plt.savefig(plots_folder + "Lenghts.png", format='png')
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

## Words frequency distribution
labels = ['1','2','3','4','5','6','7','8','9','10','>10']
b1 = plt.bar(np.arange(11), words_frequency_distribution, tick_label=labels)
plt.gca().set(title="Words Frequency Distribution", ylabel="Words", xlabel="Frequency", axisbelow=True)
autolabel(plt.gca(), b1, words_frequency_distribution/len(dictionary)*100)
plt.grid(True, linewidth=0.5)
plt.tight_layout()
plt.savefig(plots_folder + "Words_frequency_distribution.png", format='png')
plt.clf()

## Out of Vocabulary words distribution
fig, axs = plt.subplots(1, 2)
pie_chart(axs[0], OOV_words_distribution, labels=["", "OOV Words"], title="OOV Words Distribution")
pie_chart(axs[1], OOV_words_frequency, explode=0, labels=["", "OOV Words"], title="OOV Words Frequency")
plt.tight_layout()
plt.savefig(plots_folder + "OOV_words_statistics.png", format='png')
plt.clf()


print("Total distinct words: " + str(len(dictionary.keys())))