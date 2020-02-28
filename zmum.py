#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#%%
raw_df = pd.read_csv('Train.csv', sep = ';')

#%%
def remove_punctuation(text):
    text_blob = TextBlob(text)
    return ' '.join(text_blob.words)

print(remove_punctuation(raw_df['opinion'].iloc[10]))
print(raw_df['opinion'].iloc[10])

#%%
def remove_numbers_symbols_stopwords(text):
    word_list = [ele for ele in text.split() if ele != 'user']
    clean_tokens = [t for t in word_list if re.match(r'[^\W\d]*$', t)]
    clean_s = ' '.join(clean_tokens)
    clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
    return clean_mess
print(remove_numbers_symbols_stopwords(remove_punctuation(raw_df['opinion'].iloc[10])))
print(raw_df['opinion'].iloc[10])

#%%
def normalization(word_list):
        lem = WordNetLemmatizer()
        normalized_words = []
        for word in word_list:
            normalized_text = lem.lemmatize(word, 'v')
            normalized_words.append(normalized_text)
        return normalized_words
    
word_list = 'I was playing with my friends with whom I used to play, when you called me yesterday'.split()
print(normalization(word_list))

#%%
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

def text_processor(text):
    return normalization(remove_numbers_symbols_stopwords(remove_punctuation(text)))

def get_text_length(text):
    return np.array([len(x) for x in text]).reshape(-1, 1)

pipeline = Pipeline([
    # ('features', FeatureUnion([
    #     ('text', Pipeline([
    #         ('vectorizer', CountVectorizer(min_df=1,max_df=2)),
    #         ('tfidf', TfidfTransformer()),
    #     ])),
    #     ('length', Pipeline([
    #         ('count', FunctionTransformer(get_text_length, validate=False)),
    #     ]))
    # ])),
    ('bow', CountVectorizer(analyzer=text_processor)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

#%%
msg_train, msg_test, label_train, label_test = train_test_split(raw_df['opinion'], raw_df['rate'], test_size=0.2)
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))

# %%
