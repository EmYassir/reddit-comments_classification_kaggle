# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:59:58 2019

@author: Yassir
"""
import csv
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2



from dictionary.dictionary import Dictionary
from text.text_util import Text_Util



def test_model(model_name, model, X_train, X_test, y_train, y_test):
    print('########## Testing ' + str(model_name) + ' ##########')
    print('Training ...')
    model.fit(X_train, y_train)
    print('Predicting ...')
    y_pred = model.predict(X_test)
    print('-> accuracy %s' % accuracy_score(y_pred, y_test))
    #print(classification_report(y_test, y_pred, target_names=np.unique(labels).tolist()))

def preprocess_text(path, test = False):
    comments = None 
    labels = None
    
    # Loading data
    if test:
        comments = np.load(path, allow_pickle=True)
    else:
        comments, labels = np.load(path, allow_pickle=True)
        
    # Preprocessing text
    text_util = Text_Util()
    X = text_util.get_preprocessed_sentences(np.array(comments))
    y = None
    if not test:
        y = np.array(labels)
    print('Total words in the corpus before cleanup: %d' 
      %(text_util.get_number_scanned_words()))
    return X, y


def create_tfidf_vector(X):
    print('Creating word count/tfidf vectors ...')
    X_tfidf = TfidfVectorizer().fit_transform(X)
    print('Tfidf vector size = (%d, %d)' %(X_tfidf.shape[0], X_tfidf.shape[1]))
    return X_tfidf

def select_features(k, X, y):
    print('Performing feature selection ...')
    print('Selecting %d best features:' %k)
    X_sel = SelectKBest(chi2, k).fit_transform(X, y)
    return X_sel

def split_data(X, y, p=0.015):
    print('Splitting data ...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.015)
    print('-> Training set (size = %d)' %len(y_train))
    print('-> Test set (size = %d)' %len(y_test))
    return X_train, X_test, y_train, y_test


def dump_predictions(y_pred, filepath="submission.csv"):
    with open(filepath, mode='w', newline='') as submission_file:
        writer = csv.writer(submission_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
        writer.writerow(["Id", "Category"])
        for i, prediction in enumerate(y_pred):
            writer.writerow([i, prediction])
