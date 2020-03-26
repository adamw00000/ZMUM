# %%
import pandas as pd
import numpy as np

df = pd.read_csv('no_abbreviations_Train.csv')

# %%
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Ridge

# %%
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
    train_test_split(df['opinion'], df['rate'], \
    test_size=0.2)

msg_train.to_csv("corpus_train.txt", index=False)
msg_test.to_csv("corpus_test.txt", index=False)

corpus_train = open('corpus_train.txt', 'r', encoding='utf8')
corpus_test = open('corpus_test.txt', 'r', encoding='utf8')

def scoresOnCV(model, wordvec, description):
    prediction = model.predict(wordvec)
    accuracy = np.mean(label_test == np.round(prediction))
    print(description + ": accuracy is "+ str(accuracy))
    RMSE = np.sqrt(np.sum((label_test - prediction)**2)/len(label_test))
    print(description + ": RMSE is "+ str(RMSE))

    return accuracy, RMSE

vectorizer = CountVectorizer(stop_words = 'english', min_df=0.001)
wordvec_train = vectorizer.fit_transform(corpus_train)
wordvec_test = vectorizer.transform(corpus_test)

print("dimension of the word count vector is (%d, %d)"\
    %(wordvec_train.shape[0], wordvec_train.shape[1]))

print('min_df =', 0.001)

mnb_clf = MultinomialNB()

mnb_clf.fit(wordvec_train, label_train)
mnb_accuracy_cv, mnb_RMSE_cv = scoresOnCV(mnb_clf, wordvec_test, "multinomial naive Bayes")

bnb_clf = BernoulliNB()

bnb_clf.fit(wordvec_train, label_train)
bnb_accuracy_cv, bnb_RMSE_cv = scoresOnCV(bnb_clf, wordvec_test, "Bernoulli naive Bayes")

# %%
corpus_train = open('corpus_train.txt', 'r', encoding='utf8')
corpus_test = open('corpus_test.txt', 'r', encoding='utf8')

vectorizer = CountVectorizer(stop_words = 'english', min_df=0.01)
wordvec_train = vectorizer.fit_transform(corpus_train)
wordvec_test = vectorizer.transform(corpus_test)

print("dimension of the word count vector is (%d, %d)"\
    %(wordvec_train.shape[0], wordvec_train.shape[1]))

print('min_df =', 0.01)

mnb_clf = MultinomialNB()

mnb_clf.fit(wordvec_train, label_train)
mnb_accuracy_cv, mnb_RMSE_cv = scoresOnCV(mnb_clf, wordvec_test, "multinomial naive Bayes")

# %%
corpus_train = open('corpus_train.txt', 'r', encoding='utf8')
corpus_test = open('corpus_test.txt', 'r', encoding='utf8')

vectorizer = CountVectorizer(stop_words = 'english', min_df=0.000005)
wordvec_train = vectorizer.fit_transform(corpus_train)
wordvec_test = vectorizer.transform(corpus_test)

print("dimension of the word count vector is (%d, %d)"\
    %(wordvec_train.shape[0], wordvec_train.shape[1]))

print('min_df =', 0.000005)

mnb_clf = MultinomialNB()

mnb_clf.fit(wordvec_train, label_train)
mnb_accuracy_cv, mnb_RMSE_cv = scoresOnCV(mnb_clf, wordvec_test, "multinomial naive Bayes")

# %%
corpus_train = open('corpus_train.txt', 'r', encoding='utf8')
corpus_test = open('corpus_test.txt', 'r', encoding='utf8')

vectorizer = CountVectorizer(stop_words = 'english', min_df=0.000005)
wordvec_train = vectorizer.fit_transform(corpus_train)
wordvec_test = vectorizer.transform(corpus_test)

logistic_clf = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=300) 
logistic_clf.fit(wordvec_train, label_train)
logi_accuracy_cv, logi_RMSE_cv = scoresOnCV(logistic_clf, wordvec_test, "multinomial logistic regression")

print("logistic regression (multinomial) training: done in %0.3fs" % (time() - t0))

# %%
