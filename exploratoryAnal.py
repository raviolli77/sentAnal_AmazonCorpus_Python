#!/usr/bin/env python3
# Load appropriate modules
import string
import re
from helperFunctions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Visuals
from wordcloud import WordCloud
import nltk
from nltk import corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

plt.style.use('ggplot')

# Set the names for the columns
names = ['ind', 'text', 'label']

# Read in CSV file
amazon = pd.read_csv('amazonLabelled.csv', names = names)

amazon.set_index(['ind'], inplace = True)

# Exploratoary analysis on entire data set
corpAmazon = amazon['text']
labels = amazon['label']

lowerCaps = lowerCaps(corpAmazon)
corp_toke = tokenizer(lowerCaps)
corp_amazon = otherShit(corp_toke)

data = list(zip(corp_amazon, labels))

print('\n')
print('Only positive word tokens:')
# List that only keeps text that has been labeled as 'Positive'
pos_tokes = [word[0] for word in data if word[1] == 'Positive']

pos_count = createHist(pos_tokes, color = 'blue', tokenName = 'Positively Labeled Corpus')

# Basic Statistics relating to corpus
print('Average length of tokens for Positively Labeled Corpus: ')
print(sum(pos_count)/len(pos_count))

print('Max token count for Positively Labeled Corpus: ')
print(max(pos_count))

print('Min token count for Positively Labeled Corpus: ')
print(min(pos_count))

# Create corpus 
pos_corpus = createCorpus(pos_tokes)

# Word Cloud
createWordcloud(pos_corpus, 'Positive Labelled Posts')

# Function that will print and graph top ten most used words in corpus
pos_corpus_top_count = topTen(pos_corpus, 'Positive Labelled Posts', color = 'blue', plotBar = True)

print('\n')
print('Most common words: ')
for key in pos_corpus_top_count:
	print(key[0], '|', key[1])

print('\n')
print('Only negative word tokens:')
# List that only keeps text that has been labeled as 'Negative'
neg_toks = [word[0] for word in data if word[1] == 'Negative']

neg_count = createHist(neg_toks, color = 'red', tokenName = 'Negatively Labeled Corpus')

# Basic Statistics relating to corpus
print('Average length of tokens for Negatively Labeled Corpus: ')
print(round(sum(neg_count)/len(neg_count), 3))

print('Max token count for Negatively Labeled Corpus: ')
print(max(neg_count))

print('Min token count for Negatively Labeled Corpus: ')
print(min(neg_count))

# Create corpus 
neg_corpus = createCorpus(neg_toks)

# Word Cloud
createWordcloud(neg_corpus, 'Negative Labelled Posts')

# Function that will print and graph top ten most used words in corpus
neg_corpus_top_count = topTen(neg_corpus, 'Negative Labelled Posts', color = 'red', plotBar = True)

print('\n')
print('Most common words: ')
for key in neg_corpus_top_count:
	print(key[0], '|', key[1])

print('\n')
print('Entire Corpus')

whole_toks = [word[0] for word in data]
whole_count = createHist(whole_toks, color = '#00868B', tokenName = 'Entire Corpus')

# Basic Statistics relating to corpus
print('Average length of tokens for Entire Corpus: ')
print(round(sum(whole_count)/len(whole_count), 3))

print('Max token count for Entire Corpus: ')
print(max(whole_count))

print('Min token count for Entire Corpus: ')
print(min(whole_count))
# Create corpus 
whole_corpus = createCorpus(corp_amazon)

# Word Cloud
createWordcloud(whole_corpus, 'Entire Corpus')

# Function that will print and graph top ten most used words in corpus
whole_corpus_top_counts = topTen(whole_corpus, 'Entire Corpus', color = '#00868B', plotBar = True)

print('\n')
print('Most common words: ')
for key in whole_corpus_top_counts:
	print(key[0], '|', key[1])