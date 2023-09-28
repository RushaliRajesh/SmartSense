import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("deceptive-opinion-spam-corpus\deceptive-opinion.csv")

'''Analysis of the data/ exploring the datset'''
# print(dataset.head())
# print(dataset.isnull().sum())
# print(dataset['deceptive'].value_counts())
# print(dataset['hotel'].value_counts())
# print(dataset['hotel'].unique())
# print(dataset['hotel'].unique().shape[0])

'''preprocessing: stopword removal, tokensization, vectorisation'''

'''1. stop words removal'''
nltk.download('stopwords')
# print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
# dataset.drop(['clean_text'], axis=1)
dataset['clean_text'] = dataset['text'].copy()
dataset['clean_text'] = dataset['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
# print(dataset['text'], dataset['clean_text'])

'''2. Tokenizing the clean_text column'''
tt = TweetTokenizer()
dataset['tokens'] = dataset['clean_text'].apply(tt.tokenize)

'''Vectorization of the clean_text column and it is the feature vector for model training'''
x = dataset['clean_text']
cv1 = CountVectorizer()
x_vect = cv1.fit_transform(x)
x_vect = x_vect.toarray()
x_vect


'''getting the labels encoded'''
# dataset['labels'] = dataset['deceptive'].copy()
# for ind,i in enumerate(dataset['deceptive']):
#   if i=='truthful':
#     dataset['labels'][ind] = int(0)
#   else:
#     dataset['labels'][ind] = int(1)
# Import label encoder
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
dataset['labels']= label_encoder.fit_transform(dataset['deceptive'])
# dataset['labels']


labels = dataset['labels'] #labels for training the model
# print(x_vect.shape, labels)
'''Training and evaluating Logistic regression model'''
x_train, x_test, y_train, y_test = train_test_split(x_vect,labels, random_state=0)
# print(y_train)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Accuracy = logreg.score(x_test, y_test)
print("Logistic regression model accuracy: ", Accuracy*100)


