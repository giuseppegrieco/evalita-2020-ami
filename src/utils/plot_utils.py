import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, median

# Add labels on rects in bar plot
def autolabel(ax, bars, bar_labels, y_pos=1.05, fontsize=9):
    for idx, rect in enumerate(bars):
        y = rect.get_height()
        x = rect.get_x() + rect.get_width()/2.
        ax.text(x, y_pos*y, r"%3.1f%%" % (bar_labels[idx]), ha='center', va='bottom', fontsize=fontsize)


# Make histplot with the distribution shape and the mean/median
def enriched_histplot(ax, values,
					  xlabel=None, ylabel=None,
					  title=None, legend=False):

    sns.histplot(values, ax=ax, kde=True)
    _, ymax = ax.get_ylim()
    ax.plot([mean(values), mean(values)], [0, 0.9*ymax], color='red', label="Mean")
    ax.plot([median(values), median(values)], [0, 0.9*ymax], color='green', label="Median")
    ax.grid(True, linewidth=0.5)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, axisbelow=True)
    if(legend): ax.legend()

#
def pie_chart(ax, values, explode=0.05, labels=None, colors=None, title=None):
	ax.pie(values, explode=[explode]*len(values), labels=labels, colors=colors, shadow=True, autopct='%1.1f%%')
	ax.set_title(title)