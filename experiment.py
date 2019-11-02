import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC

from dictionary.dictionary import Dictionary
from text.text_util import Text_Util


# Loading data
train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'
comments, labels = np.load(train_set_path, allow_pickle=True)


# Preprocessing text
print('### Preprocessing text...')
text_util = Text_Util()
X = text_util.old_get_preprocessed_sentences(np.array(comments))
y = np.array(labels)
print('Total words in the corpus before cleanup: %d' 
      %(text_util.get_number_scanned_words()))


# Experiment with countVectorizer + TfidfTransformer
print('### Begin experimentation')
print('Creating word count/tfidf vectors ...')
#X_counts = CountVectorizer().fit_transform(X)
#X_tfidf = TfidfTransformer().fit_transform(X_counts)
X_tfidf = TfidfVectorizer().fit_transform(X)
print('Tfidf vector size = (%d, %d)' %(X_tfidf.shape[0], X_tfidf.shape[1]))

print('Performing feature selection ...')
k = 10000
print('Selecting %d best features:' %k)
X_sel = SelectKBest(chi2, k).fit_transform(X_tfidf, y)
#X_sel =  X_tfidf

print('Splitting data ...')
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, stratify=y, test_size=0.015)
print('-> Training set (size = %d)' %len(y_train))
print('-> Test set (size = %d)' %len(y_test))

print('########## Testing Multinomial Naive Bayes ##########')
print('Training ...')
model = MultinomialNB()
model.fit(X_train, y_train)

print('Predicting ...')
y_pred = model.predict(X_test)

print('-> accuracy %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred, target_names=np.unique(labels).tolist()))

print('########## Testing Logistic Regression ##########')
print('Training ...')
model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter = 1000)
model.fit(X_train, y_train)

print('Predicting ...')
y_pred = model.predict(X_test)
print('-> accuracy %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred, target_names=np.unique(labels).tolist()))

print('########## Testing Support Vector Machines ##########')
print('Training ...')
model = LinearSVC()
model.fit(X_train, y_train)

print('Predicting ...')
y_pred = model.predict(X_test)
print('-> accuracy %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred, target_names=np.unique(labels).tolist()))

print('########## Testing Support Vector Machines with SGD ##########')
print('Training ...')
model = SGDClassifier()
model.fit(X_train, y_train)

print('Predicting ...')
y_pred = model.predict(X_test)
print('-> accuracy %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred, target_names=np.unique(labels).tolist()))

print('########## Testing Random Forests ##########')
print('Training ...')
model = RandomForestClassifier(n_estimators=2000, max_depth=10, random_state=0)
model.fit(X_train, y_train)

print('Predicting ...')
y_pred = model.predict(X_test)

print('-> accuracy %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred, target_names=np.unique(labels).tolist()))



    