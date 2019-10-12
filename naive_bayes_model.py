import numpy as np
import pickle as pkl
from text_processing_util import TextProcessingUtil as tpu

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
        self._X = None
        self._y = None
        self._classes = None
        self._mapping = {}
        self._wordProbVec = None
        self._classProbVec = None

    def load(self, inFilePath = 'model.npy'):
        pass
    
    def dump(self, outFilePath = 'model.npy'):
        pass

    def train(self, tokenized_sentence):
        # Getting preprocessed data
        util = tpu.TextProcessingUtil(tokenized_sentence)
        word_matrix = util.get_bow_matrix(tokenized_sentence)
        self._X = word_matrix[:,0:-1]
        self._y = word_matrix[:,-1]
        # Getting classes
        self._classes = np.sort(np.unique(self._y))
        
        # Mapping vocabulary
        vocab_sorted = util.getSortedVocabulary()
        for i, v in enumerate(vocab_sorted):
            self._mapping[v] = i
            
        # Computing classes probabilities
        self._classProbVec = np.zeros(self._classes.shape)
        for i,c in enumerate(self._classes):
            self._classProbVec[i] = np.mean(self._y == c)
            
        # Computing words likelihoods
        self._wordProbVec = np.zeros(vocab_sorted.shape[0], self._classes.shape[0])
        for k,v in self._mapping:
            for i,c in enumerate(self._classes):
                x = self._X
                nominator = np.sum(x[x[self._y == c], v], axis = 0) + self._alpha
                denominator = np.sum(x[:,v], axis = 0) + self._alpha * vocab_sorted.shape[0]
                self._wordProbVec[v, i] = nominator / denominator

        
    # Returns the class to which the sentence belongs 
    def predict(self, tokenized_sentence):
        predictions = np.zeros(self._classProbVec.shape)
        words = np.split(tokenized_sentence)
        for c in np.arg(self._classes):
            p = 1.0
            for w in words:
                if w in self._mapping:
                    p *= self._wordProbVec[self._mapping[str(w)]]
                else:
                    p *= (1.0 + self._alpha) / (1.0 + self._alpha  * len(self._mapping)) 
            predictions[c] = p * self._classProbVec[c]
        return np.max(predictions)
            
            
        
            
        
        