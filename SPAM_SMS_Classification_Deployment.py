
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 23:04:53 2020

@author: 97156
"""


import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv("Spam SMS Collection", sep="\t", names=['label', 'message'])

# mapping values for the label
df["label"] = df["label"].map({'ham': 0, 'spam': 1})

# handling imbalanced data with oversampling
spam_data = df[df['label'] == 1]
count = int((df.shape[0] - spam_data.shape[0]) / spam_data.shape[0])
for i in range(0, count - 1):
    df = pd.concat([df, spam_data])

# Importing essential Libraries for NLP and data cleaning
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# cleaning the message
corpus = []
wnl = WordNetLemmatizer()

for sms_string in list(df['message']):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms_string)
    message = message.lower()
    words = message.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    words = [wnl.lemmatize(word) for word in words]
    message = ' '.join(words)
    corpus.append(message)

# Creating the Bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names()

X = pd.DataFrame(vectors, columns=feature_names)
y = df['label']

# Creating a pickle file for the CountVectorizer
pickle.dump(tfidf, open('tfidf-transform.pkl', 'wb'))

# Model Building
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Fitting Random Forest to train data
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)

# Creating a pickle file for Random Forest Model
pickle.dump(classifier, open('spam-sms-rfc-model.pkl', 'wb'))
