from classifier import preprocess_text
from classifier import create_tfidf_vector
from classifier import select_features
from classifier import split_data
from classifier import test_model
from classifier import dump_predictions

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import csv

# Loading data
train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'
output_file = './output/submission.csv'
comments, labels = np.load(train_set_path, allow_pickle=True)


# Experiment with countVectorizer + TfidfTransformer
print('### Begin program')
print('Processing train data')
X, y = preprocess_text(train_set_path, False)

# Creating words vectors
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print('Tfidf vector size = (%d, %d)' %(X.shape[0], X.shape[1]))

# Selecting best features
#k = 25000
#print('Performing feature selection ...')
#print('Selecting %d best features:' %k)
#Selector = SelectKBest(chi2, k).fit(X, y)
#indexes = Selector.get_support()
#print(indexes)
#print(indexes.shape)
#X_sel = SelectKBest(chi2, k).fit_transform(X, y)
#Xsel = select_features(k, X, y)

# Instantiating models
X_train, X_test, y_train, y_test = split_data(X, y, 0.015)
alphas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 1.0]
alpha_star = 0.0
best_score = 0.0
for alpha in alphas:
    model = MultinomialNB(alpha)
    print('Training model ...')
    model.fit(X_train, y_train)
    print('Predicting ...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    print('-> alpha = %f : accuracy %s' %(alpha, accuracy))
    if accuracy > best_score:
        best_score = accuracy
        alpha_star = alpha


print('Re-training model on the whole data ...')
model = MultinomialNB(alpha_star)
model.fit(X, y)

print('Processing test data')
X, y = preprocess_text(test_set_path, True)
X = vectorizer.transform(X)
print('Predicting ...')
#y_pred = model.predict(X[:,np.array(indexes)])
y_pred = model.predict(X)
print('Generating file ...')
dump_predictions(y_pred, output_file)
    
   
print('### End of program')



    