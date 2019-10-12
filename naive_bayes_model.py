import numpy as np
import pickle as pkl
from text_processing_util import TextProcessingUtil

'''
class TextProcessingUtil:

    def _clean_urls(self, sentence):
  
    def _remove_stop_words(self, tokenized_sentence):

    def _apply_lemmatizer(self, tokenized_sentence):


    def _preprocess_sentence(self, sentence):


    def _get_bow_representation(self, tokenized_sentence):

    def _get_numeric_labels(self, label_values):

    def get_bow_matrix(self, sentences, label_values=None):
 '''
 
class NaiveBayesWithSmoothing:

    def __init__(self, alpha = 0.0, filename = 'model.npy'):
        self._alpha = alpha
        self._filename = filename
        self._util = TextProcessingUtil()
        self._X = None
        self._y = None
        self._classes = None
        self._wordProbVec = None
        self._classProbVec = None
        
    def load(self, inFilePath = 'model.npy'):
        pass
    
    def dump(self, outFilePath = 'model.npy'):
        pass

    def get_classes():
        return self._classes
    
    def train(self, sentences, labels):
        # Getting preprocessed data
        word_matrix = self._util.get_bow_matrix(sentences, labels)
        self._X = word_matrix[:,0:-1]
        self._y = word_matrix[:,-1]
        d = self._X.shape[1]
        # Getting classes
        self._classes = self._util.get_labels()
        
        # Computing classes probabilities
        self._classProbVec = np.zeros(self._classes.shape)
        for i,c in enumerate(self._classes):
            y = self._y
            self._classProbVec[i] = np.mean(y==c)
            
        # Computing words likelihoods
        self._wordProbVec = np.zeros((d, self._classes.shape[0]))
        for v in range(d):
            for i,c in enumerate(self._classes):
                x = self._X
                y = self._y
                nominator = np.sum(np.where(y == c)[0], axis = 0) + self._alpha
                denominator = np.sum(x[:,v], axis = 0) + self._alpha * d
                self._wordProbVec[v, i] = nominator / denominator

        
    # Returns the class to which the sentence belongs 
    def predict(self, sentences):
        bag_of_words = self._util.get_bow_matrix(sentences)
        n = bag_of_words.shape[0]
        d = bag_of_words.shape[1]
        predictions = np.zeros((bag_of_words.shape[0], self._classes.shape[0]))
        for i in range(n):
            for k, c in enumerate(self._classes):
                p = 1.0
                for j in range(d):
                    p *= self._wordProbVec[j, k] 
                predictions[i, k] = p * self._classProbVec[k]
        
        return np.max(predictions, axis = 1)
            
            
        
            
        
        