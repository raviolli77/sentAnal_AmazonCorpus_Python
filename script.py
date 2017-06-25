#!/usr/bin/env python3
# Load appropriate modules
import string
import re
from helperFunctions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Visuals
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

# Set the names for the columns
names = ['ind', 'text', 'label']

# Read in CSV file
amazon = pd.read_csv('amazonLabelled.csv', names = names)

amazon.set_index(['ind'], inplace = True)

print('\n')
print('Count for labels:')
print(amazon['label'].value_counts())

# Print the data frame 
print('\n')
print(amazon.head())

# Exploratory analysis 
print('Describe function here')
print(amazon['text'].describe())

# Creating training and test split
train, test = train_test_split(amazon, 
	test_size = 0.20, 
	random_state = 42)

# First we lower caps everything with list comprehensions
train_text = train['text']
train_class = train['label'].tolist()

test_text = test['text']
test_class = test['label'].tolist()



# Lower caps
train_lowerCaps = lowerCaps(train_text)
test_lowerCaps = lowerCaps(test_text)

# Next we tokenize
train_toke = tokenizer(train_lowerCaps)
test_toke = tokenizer(test_lowerCaps)

# Rest of stuff
last_train = otherShit(train_toke)
last_test = otherShit(test_toke)

# Might need
#data = list(zip(rmvSpcChar, labels))

#print(data)

tfidf_vect = TfidfVectorizer(tokenizer=lambda doc: doc, 
	lowercase=False, 
	stop_words = 'english',
	ngram_range=(1, 3))

# Convert our training set to tfidf
train_words = np.array(last_train)
train_counts = tfidf_vect.fit_transform(train_words)

# Convert our test set to tfidf
test_words = np.array(last_test)
test_counts = tfidf_vect.transform(test_words)

# Fit model 
model1 = MultinomialNB()

# Train model
model1.fit(train_counts, train_class)

# Predict using test set
predictions = model1.predict(test_counts)
##
#print(len(predictions))
#print(len(test_class))
##
accuracy = model1.score(test_counts, test_class)
##
test_error_rate = 1 - accuracy
##

print('\n')
print('Test Set Error Rate:')
print(np.mean(predictions != test_class))

# Converting to series to be able to use the cross tab functionality in python
test_class_series = pd.Series(test_class)
predictions_series = pd.Series(predictions)

print('\n')
print('Confusion matrix:')
print(pd.crosstab(predictions_series, test_class_series,
	rownames=['Predicted Values'], 
	colnames=['Actual Values']))

data = pd.DataFrame(list(zip(test_text, test_class, predictions_series)), 
	columns = ['text', 'label', 'predictions'])

# print(data.head())
print('\n')
print("Classification Report for test set:")

print(classification_report(predictions_series, test_class_series))

# data.to_csv('results.csv', index = False)

predictions_num = predictions_series.map({'Positive':1, 'Negative':0})

test_class_num = test_class_series.map({'Positive':1, 'Negative':0})

fpr, tpr, _ = roc_curve(predictions_num, 
	test_class_num)

auc_nb = auc(fpr, tpr)

f, ax = plt.subplots(figsize=(8, 10))

plt.plot(fpr, tpr, 
	label = 'Naive Bayes',
	color='navy',
	linewidth=1)

ax.set_facecolor('#fafafa')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison For Naive Bayes Model (AUC: {0})'\
	.format(round(auc_nb, 3)))
plt.legend(loc="lower right")
	
plt.show()

print("Fin :)")