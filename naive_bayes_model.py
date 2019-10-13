import numpy as np
import pickle as pkl
from text_processing_util import TextProcessingUtil

class NaiveBayesWithSmoothing:
    
    def __init__(self, alpha=0.0, filename='model.npy'):
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

    def get_classes(self):
        return self._classes
    
    def log_prob(self, p):
        if p != 0.0:
            return np.log(p)
        return 0.0
        
    
    def train(self, sentences, labels):
        print('train')
        # Getting preprocessed data
        word_matrix = self._util.get_bow_matrix(sentences, labels)
        self._X = word_matrix[:,0:-1]
        self._y = word_matrix[:,-1]
        n, d = self._X.shape
        # Getting classes
        self._classes = self._util.get_labels()
        
        # Computing classes probabilities
        self._classProbVec = np.zeros(self._classes.shape)
        for i, c in enumerate(self._classes):
            c_idx = np.where(self._y == i)[0]
            self._classProbVec[i] = self.log_prob(c_idx.shape[0]) - self.log_prob(n)
            
        # Computing words likelihoods
        print('processing words...')
        self._wordProbVec = np.zeros((d, self._classes.shape[0]))
        for v in range(d):
            for i, c in enumerate(self._classes):
                x = self._X
                c_idx = np.where(self._y == i)[0]
                numerator = np.sum(self._X[c_idx, v], axis =0) + self._alpha
                denominator = np.sum(self._X[:,v], axis =0) + self._alpha * d
                self._wordProbVec[v, i] = self.log_prob(numerator) - self.log_prob(denominator)


    # Returns the class to which the sentence belongs 
    def predict(self, sentences):
        print('predict')
        bag_of_words = self._util.get_bow_matrix(sentences)
        n = bag_of_words.shape[0]
        predictions = np.zeros((n, self._classes.shape[0]))
        print('processing sentences...')
        for i, test_sentence in enumerate(bag_of_words):
            for k, c in enumerate(self._classes):
                p = 0.0
                idx_list = np.where(test_sentence != 0)[0]  # TODO handle the case where no words in the voc are present
                for j in idx_list:
                    p += self._wordProbVec[j, k] 
                predictions[i, k] = p + self._classProbVec[k]
        
        return np.argmax(predictions, axis=1)
    