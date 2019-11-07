# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:59:58 2019

@author: Yassir
"""
import csv
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report




from dictionary.dictionary import Dictionary
from text.text_util import Text_Util


class Classifier_Util:
    
    # Init
    def __init__(self):
        self._feat_selector = None
        self._word_vectorizer = None
        self._voting_classifier = None
    
    
    def set_word_vectorizer(self, model):
        self._word_vectorizer = model
    
    def set_feature_selector(self, model):
        self._feat_selector = model
    
    def set_voting_classifier(self, models, voting_classifier):
        estimators = []
        for k, v in models.items():
            estimators.append((k, v))
        self._voting_classifier = voting_classifier(estimators)
    
    
    def preprocess_text(self, path, test = False):
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
        y = np.empty((X.shape[0],))
        if test == False:
            y = np.array(labels)
        print('Total words in the corpus before cleanup: %d' 
          %(text_util.get_number_scanned_words()))
        
        # Debug
        output_path = './output/'
        if test:
            output_path += 'dump_test.txt'
        else:
            output_path += 'dump_train.txt'
        text_util.dump_data(X, output_path)
        return X, y
    
    # Removes words shared between all labels
    def preprocess_text_with_removal(self, path, test = False):
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
        
        # Total number
        number = text_util.get_number_scanned_words()
        print('Total words in the corpus before cleanup: %d' %number)
        
        # No further processing for tests
        if test:
            return X, y
        
        # Updating the dictionary
        print('Updating the dictionary...')
        y = np.array(labels)
        dic = Dictionary(np.unique(y))
        for i, sentence in enumerate(X):
            dic.update_sentence(sentence, y[i])
        print('Total words in the corpus after cleanup: %d' %dic.get_global_count())
        
        new_X = np.copy(X)
        words_in_common = dic.get_n_words_common_to_labels()
        print('Filtering %d words' %len(words_in_common))
        for i, sentence in enumerate(X):
            new_X[i] = text_util.filter_sentence(sentence, words_in_common)
        
        return new_X, y
    
    
    def fit_words_vectorizer(self, X, y):
        self._word_vectorizer.fit(X, y)
        
    def get_words_vector(self, X):
        return self._word_vectorizer.transform(X)
    
    def fit_feature_selector(self, X, y):
        self._feat_selector.fit(X, y)
        
    def get_selected_features(self, X):
        return self._feat_selector.transform(X)
    
    def split_data(self, X, y, p=0.015):
        print('Splitting data ...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.015)
        print('-> Training set (size = %d)' %len(y_train))
        print('-> Test set (size = %d)' %len(y_test))
        return X_train, X_test, y_train, y_test
    
    
    def dump_predictions(self, y_pred, filepath="submission.csv"):
        with open(filepath, mode='w', newline='') as submission_file:
            writer = csv.writer(submission_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            writer.writerow(["Id", "Category"])
            for i, prediction in enumerate(y_pred):
                writer.writerow([i, prediction])
    
    def test_voting_models(self, X_train, X_test, y_train, y_test):
        print('########## Testing voting model ##########')
        print('Training ...')
        self._voting_classifier.fit(X_train, y_train)
        print('Predicting ...')
        
        y_pred = self._voting_classifier.predict(X_train)
        print('-> accuracy on train %s' % accuracy_score(y_pred, y_train))
        
        y_pred = self._voting_classifier.predict(X_test)
        print('-> accuracy on test %s' % accuracy_score(y_pred, y_test))
        
        txt_util = Text_Util()
        txt_util.dump_data(y_pred, './output/y_pred.txt')
        txt_util.dump_data(y_test, './output/y_test.txt')
        #print(classification_report(y_test, y_pred, target_names=np.unique(labels).tolist()))
        
    def train_predict_dump(self, X_train, y_train, X_test, model = None):
        print('########## Final program ##########')
        if model is None:
            model = self._voting_classifier
        
        print('Training ...')
        model.fit(X_train, y_train)
        print('Predicting ...')
        
        y_pred = self._voting_classifier.predict(X_train)
        print('-> accuracy on train %s' % accuracy_score(y_pred, y_train))
        y_pred = self._voting_classifier.predict(X_test)
        self.dump_predictions(y_pred)
