#!/usr/bin/env python3
# Load appropriate modules
import string
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Visuals
from wordcloud import WordCloud
import nltk
from nltk import corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import auc
# nltk.download('punkt')
# nltk.download('stopwords')

def lowerCaps(corpus): 
	thing = [word.lower() for word in corpus]
	return thing

def tokenizer(corpus):
	tokens = [word_tokenize(word) for word in corpus]
	return tokens

def otherShit(corpus):
	exclude = set(string.punctuation)
	myexclude = ["'s", "n't", "'ve", 'would']
	rmvPunc = [[el for el in word if el not in exclude and el not in myexclude] for word in corpus]
	rmvStpWrds = [[el for el in word if el not in stopwords.words('english')] for word in rmvPunc]
	rmvNum = [[el for el in word if not el.isdigit()] for word in rmvStpWrds]
	rmvSpcChar = [[el for el in word if re.sub('[^A-Za-z0-9]+', '', el)] for word in rmvNum]
	return rmvSpcChar

def createCorpus(corpus):
	newList = []
	for word in corpus:
		for otherWord in word:
			newList.append(otherWord)
	return newList

def createHist(token, color, tokenName):
	toke_count = [len(word[0]) for word in token]
	
	f, ax = plt.subplots(figsize=(11, 11))		
	ax.set_facecolor('#fafafa')
	
	plt.hist(toke_count,
		color=color,
		edgecolor='white',
		alpha = 0.75)
	plt.title('Token Lengths for {0}'.format(tokenName))
	plt.xticks(np.arange(0, max(toke_count)+1, 1.0))
	plt.xlabel('Token Count per Entry')
	plt.ylabel('Frequency')
	
	plt.show()
	return toke_count

def createWordcloud(corpus, corpusName):
	# Generate a word cloud image
	corp_str = ' '.join(corpus)
	wordcloud = WordCloud().generate(corp_str)

	plt.imshow(wordcloud, interpolation='bilinear')
	plt.title('Word Cloud for {0}'.format(corpusName))
	plt.axis("off")

	plt.show()


def topTen(corpus, corpusName, color, plotBar = True):
	corpus_counts = Counter(corpus)
	topTen = corpus_counts.most_common(10)
	dict_counts = dict(topTen)

	words = sorted(dict_counts, key = dict_counts.__getitem__, reverse = True)
	counts = [value for (key, value) in dict_counts.items()]
	counts = sorted(counts, reverse=True)

	dict_test = list(zip(words, counts))
	if plotBar:	
		y_pos = np.arange(len(words))	
		
		f, ax = plt.subplots(figsize=(11, 11))		
		ax.set_facecolor('#fafafa')
		plt.bar(y_pos, counts,
			color = '{0}'.format(color))
		plt.xticks(y_pos, words)
		plt.ylabel('Counts')
		plt.title('Most Common words in {0}'.format(corpusName))	
		plt.show()
	return dict_test