# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:57:40 2019

@author: Yassir
"""
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# Downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


class NlpUtil:
    # Init
    def __init__(self):
        self._lemmatizer = WordNetLemmatizer()
        self._stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self._stop_words = set(stopwords.words('english'))
        self._tokenizer = RegexpTokenizer(r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S')
        self._banned = {'www': 1, 'http': 2}
    
    def _lower_chars(self, comment):
        return comment.lower()
    
    
    def _remove_stop_words(self, tokenized_comment):
        return [w for w in tokenized_comment if w not in self._stop_words]
    
    
    def _remove_non_alpha(self, tokenized_comment):
        alphanums = []
        for token in tokenized_comment:
            try:
                is_alpha = token.encode('ascii').isalpha()
            except UnicodeEncodeError:
                is_alpha = False
            
            if is_alpha == True:
                alphanums.append(token)
        return alphanums
    '''
    def _remove_non_alpha(self, tokenized_comment):
        alphanums = []
        for token in tokenized_comment:
            try:
                float(token)
            except ValueError:
                alphanums.append(token)
        return alphanums
    '''
        
    def _remove_urls(self, tokenized_comment):
        words = []
        for word in tokenized_comment:
            if word not in self._banned:
                words.append(word)
        return words

    
    def _remove_small_words(self, tokenized_comment):
        words = []
        for token in tokenized_comment:
            if 1 < len(token):
                words.append(token)
        return words
    
    def _lemmatize(self, tokenized_comment):
        lemmas = []
        for token in tokenized_comment:
            lemma = self._lemmatizer.lemmatize(token, 'n')
            lemmas.append(lemma)
        return lemmas
    
    def _stem(self, tokenized_comment):
        stems = []
        for token in tokenized_comment:
            stem = self._stemmer.stem(token)
            stems.append(stem)
        return stems
    
    def get_preprocessed_data(self, data):
        # 'data' is an array of comments
        n = data.shape[0]
        results = []
        for i in range(n):
            comment = data[i]   
            comment = comment.replace('_', ' ')
            aux = self._tokenizer.tokenize(comment.lower())           
            aux = self._remove_stop_words(aux)   # step1
            aux = self._remove_urls(aux)         # step2
            aux = self._remove_non_alpha(aux)    # step3
            aux = self._lemmatize(aux)           # step4
            aux = self._remove_small_words(aux)  # step5
            aux = self._stem(aux)                # step6          
            results.append(aux)
        return np.array(results)
        
        
    
    
    