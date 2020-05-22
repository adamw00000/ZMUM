# %%
import pandas as pd
import numpy as np

df = pd.read_csv('lemmed_Train.csv')
df['name'] = df['name'].astype("category").cat.codes.values
df['condition'] = df['condition'].astype("category").cat.codes.values

# %%
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge

from scipy.sparse import hstack

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


msg_train, msg_test, label_train, label_test = \
    train_test_split(df[['name', 'condition', 'opinion']], df['rate'], \
    test_size=0.2)

def rate_to_rate1(labels):
    labels = labels.astype(int)
    return np.where(np.isin(labels, [1, 2, 3]), 
                    'low',
                    np.where(np.isin(labels, [4, 5, 6, 7]), 
                            'medium',
                            'high'
                    )
            )

msg_train.opinion.to_csv("corpus_train.txt", index=False)
msg_test.opinion.to_csv("corpus_test.txt", index=False)

corpus_train = open('corpus_train.txt', 'r', encoding='utf8')
corpus_test = open('corpus_test.txt', 'r', encoding='utf8')

def scoresOnCV(model, wordvec, description):
    prediction = model.predict(wordvec)
    accuracy = np.mean(label_test == np.round(prediction))
    print(description + ": accuracy is "+ str(accuracy))

    MAE = np.sqrt(mean_absolute_error(label_test, prediction))
    print(description + ": MAE is "+ str(MAE))

    rate1_test = rate_to_rate1(label_test)
    rate1_prediction = rate_to_rate1(prediction)
    accuracy2 = np.mean(rate1_test == rate1_prediction)
    print(description + ": rate1 accuracy is "+ str(accuracy2))

    return accuracy, MAE

vectorizer = CountVectorizer(stop_words = 'english', min_df=0.001)
wordvec_train = vectorizer.fit_transform(corpus_train)
wordvec_test = vectorizer.transform(corpus_test)
# wordvec_train = hstack((wordvec_train, \
#         msg_train.condition.values[:,None], \
#         msg_train.name.values[:,None]))
# wordvec_test = hstack((wordvec_test, \
#         msg_test.condition.values[:,None], \
#         msg_test.name.values[:,None]))

print("dimension of the word count vector is (%d, %d)"\
    %(wordvec_train.shape[0], wordvec_train.shape[1]))

print('min_df =', 0.001)

mnb_clf = MultinomialNB()

mnb_clf.fit(wordvec_train, label_train)
mnb_accuracy_cv, mnb_RMSE_cv = scoresOnCV(mnb_clf, wordvec_test, "multinomial naive Bayes")


# mnb_clf = GaussianNB()

# mnb_clf.fit(wordvec_train.toarray(), label_train)
# mnb_accuracy_cv, mnb_RMSE_cv = scoresOnCV(mnb_clf, wordvec_test.toarray(), "Gaussian naive Bayes")


bnb_clf = BernoulliNB()

bnb_clf.fit(wordvec_train, label_train)
bnb_accuracy_cv, bnb_RMSE_cv = scoresOnCV(bnb_clf, wordvec_test, "Bernoulli naive Bayes")

# %%
wordvec_train

# %%
from sklearn.decomposition import TruncatedSVD

pca = TruncatedSVD(n_components = 100)
aaa = pca.fit_transform(wordvec_train, label_train)
bbb = pca.transform(wordvec_test)

aaa.shape

bnb_clf = BernoulliNB()

bnb_clf.fit(aaa, label_train)
bnb_accuracy_cv, bnb_RMSE_cv = scoresOnCV(bnb_clf, bbb, "Bernoulli naive Bayes")

# %%
for i in range(6, 7):
    min_df = 1 / 10**i
    corpus_train = open('corpus_train.txt', 'r', encoding='utf8')
    corpus_test = open('corpus_test.txt', 'r', encoding='utf8')

    vectorizer = CountVectorizer(stop_words = 'english', min_df=min_df)
    wordvec_train = vectorizer.fit_transform(corpus_train)
    wordvec_test = vectorizer.transform(corpus_test)
    # wordvec_train = hstack((wordvec_train, \
    #         msg_train.condition.values[:,None], \
    #         msg_train.name.values[:,None]))
    # wordvec_test = hstack((wordvec_test, \
    #         msg_test.condition.values[:,None], \
    #         msg_test.name.values[:,None]))

    print("dimension of the word count vector is (%d, %d)"\
        %(wordvec_train.shape[0], wordvec_train.shape[1]))

    print('min_df =', min_df)

    mnb_clf = BernoulliNB()

    mnb_clf.fit(wordvec_train, label_train)
    mnb_accuracy_cv, mnb_RMSE_cv = scoresOnCV(mnb_clf, wordvec_test, "multinomial naive Bayes")

# %%
min_df = 5e-6
corpus_train = open('corpus_train.txt', 'r', encoding='utf8')
corpus_test = open('corpus_test.txt', 'r', encoding='utf8')

vectorizer = CountVectorizer(stop_words = 'english', min_df=min_df)
wordvec_train = vectorizer.fit_transform(corpus_train)
wordvec_test = vectorizer.transform(corpus_test)
# wordvec_train = hstack((wordvec_train, \
#         msg_train.condition.values[:,None], \
#         msg_train.name.values[:,None]))
# wordvec_test = hstack((wordvec_test, \
#         msg_test.condition.values[:,None], \
#         msg_test.name.values[:,None]))

print("dimension of the word count vector is (%d, %d)"\
    %(wordvec_train.shape[0], wordvec_train.shape[1]))

print('min_df =', min_df)

mnb_clf = BernoulliNB()

mnb_clf.fit(wordvec_train, label_train)
mnb_accuracy_cv, mnb_RMSE_cv = scoresOnCV(mnb_clf, wordvec_test, "multinomial naive Bayes")

# %%
def special_round(y):
  y_rounded = np.round(y)
  y_rounded = np.where(y_rounded < 1, 0, np.where(y_rounded>10, 10, y_rounded))
  return y_rounded

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_predicted = mnb_clf.predict(wordvec_test)
# np.set_printoptions(precision=5)
cm = confusion_matrix(label_test, special_round(y_predicted), labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
cm = cm/np.sum(cm, axis=1)
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(cm, annot=True, ax = ax, fmt='d'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Normalized Confusion Matrix'); 
ax.xaxis.set_ticklabels(['1', '2','3', '4','5', '6','7', '8','9', '10']); ax.yaxis.set_ticklabels(['1', '2','3', '4','5', '6','7', '8','9', '10']);

plt.savefig('log_cm_norm.png', format = 'png', dpi = 150, bbox_inches = 'tight')

# %%
corpus_train = open('corpus_train.txt', 'r', encoding='utf8')
corpus_test = open('corpus_test.txt', 'r', encoding='utf8')

vectorizer = CountVectorizer(stop_words = 'english', min_df=0.0000005)
wordvec_train = vectorizer.fit_transform(corpus_train)
wordvec_test = vectorizer.transform(corpus_test)
# wordvec_train = hstack((wordvec_train, \
#         msg_train.condition.values[:,None], \
#         msg_train.name.values[:,None]))
# wordvec_test = hstack((wordvec_test, \
#         msg_test.condition.values[:,None], \
#         msg_test.name.values[:,None]))

t0 = time()

logistic_clf = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=300) 
logistic_clf.fit(wordvec_train, label_train)
logi_accuracy_cv, logi_RMSE_cv = scoresOnCV(logistic_clf, wordvec_test, "multinomial logistic regression")

print("logistic regression (multinomial) training: done in %0.3fs" % (time() - t0))

# %%
def special_round(y):
  y_rounded = np.round(y)
  y_rounded = np.where(y_rounded < 1, 0, np.where(y_rounded>10, 10, y_rounded))
  return y_rounded.astype(int)

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_predicted = logistic_clf.predict(wordvec_test)
# np.set_printoptions(precision=5)
cm = confusion_matrix(label_test, special_round(y_predicted), labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
cm = cm/np.sum(cm, axis=1)
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Normalized Confusion Matrix'); 
ax.xaxis.set_ticklabels(['1', '2','3', '4','5', '6','7', '8','9', '10']); ax.yaxis.set_ticklabels(['1', '2','3', '4','5', '6','7', '8','9', '10']);

plt.savefig('log_cm_norm.png', format = 'png', dpi = 150, bbox_inches = 'tight')


# %%
corpus_train = open('corpus_train.txt', 'r', encoding='utf8')
corpus_test = open('corpus_test.txt', 'r', encoding='utf8')

vectorizer = CountVectorizer(stop_words = 'english', min_df=0.0000005)
wordvec_train = vectorizer.fit_transform(corpus_train)
wordvec_test = vectorizer.transform(corpus_test)
# wordvec_train = hstack((wordvec_train, \
#         msg_train.condition.values[:,None], \
#         msg_train.name.values[:,None]))
# wordvec_test = hstack((wordvec_test, \
#         msg_test.condition.values[:,None], \
#         msg_test.name.values[:,None]))

t0 = time()

logistic_clf = LogisticRegression(multi_class='ovr', solver='sag', max_iter=300) 
logistic_clf.fit(wordvec_train, label_train)
logi_accuracy_cv, logi_RMSE_cv = scoresOnCV(logistic_clf, wordvec_test, "multinomial logistic regression")

print("logistic regression (ovr) training: done in %0.3fs" % (time() - t0))

# %%
t0 = time()

linear_reg = Ridge(alpha=0.1)
linear_reg.fit(wordvec_train, label_train)
lreg_accuracy_cv, lreg_RMSE_cv = scoresOnCV(linear_reg, wordvec_test, "linear regression")

print("linear regression: done in %0.3fs" % (time() - t0))

# %%
from sklearn.linear_model import Lasso
t0 = time()

linear_reg = Lasso(alpha=0.1)
linear_reg.fit(wordvec_train, label_train)
lreg_accuracy_cv, lreg_RMSE_cv = scoresOnCV(linear_reg, wordvec_test, "linear regression")

print("linear regression: done in %0.3fs" % (time() - t0))

# %%
