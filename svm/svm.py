import numpy as np
from dictionary import Dictionary

class NaiveBayesWithSmoothing:
    
    def __init__(self, *hyperparameters):
        argCount = len(hyperparameters)
        if argCount == 0:
            self._alpha = 0.0
        else:
            self._alpha = float((hyperparameters[0])[0])
        self._classes = None
        self._wordProbs = None
        self._classProbs= None
                
    # Getter for hyperparameter
    def get_alpha(self):
        return self._alpha
    
    # Getter for classes
    def get_classes(self):
        return self._classes
    
    # Setter for hyperparameter
    def set_alpha(self, alpha):
        self._alpha = alpha
    
    # Wrapper for np.log (handles 0)
    def _log_prob(self, p):
        if p != 0.0:
            return np.log(p)
        return 0.0

    # Returns log(p(w | c))
    def _get_log_likelihood(self, word, c):
        if word in self._wordProbs[c]:
            return (self._wordProbs[c])[word]
        return self._log_prob(self._alpha) - self._log_prob(self._V * self._alpha)
    
    # Training function
    def train(self, dico, X, y):
        # Assumes X are data and y labels....
        self._classes = dico.get_classes()
        self._classProbs = np.zeros(self._classes.shape)
        self._wordProbs =  np.array([dict() for i in range(self._classes.shape[0])])
        self._V = dico.get_global_count()
        
        # Computing classes log probabilities
        for i, c in enumerate(self._classes):
            self._classProbs[i] = self._log_prob(np.mean(y == c))
        
        # Computing log probabilities
        for i, lw in enumerate(X):
            # Loop over classes
            for c in self._classes:
                # X is an array of lists of words
                for word in lw:
                    numerator = dico.get_word_count_per_class(word, c) + self._alpha
                    denominator = dico.get_total_count_per_class(c) + self._alpha * self._V
                    (self._wordProbs[c])[word] = self._log_prob(numerator) - self._log_prob(denominator) 
    

    # Returns the class to which the sentence belongs 
    def predict(self, X):
        predictions = np.zeros(X.shape)
        buffer = np.zeros(self._classes.shape)
        
        # X is an array of lists of preprocessed words
        for i, l_w in enumerate(X):
            for c in self._classes:
                # Computing log probability
                p = 0.0
                for w in l_w:
                    p += self._get_log_likelihood(w, c)
                
                # Updating the corresponding class
                buffer[c] = p + self._classProbs[c]
                
            # Updating the prediction for comment i
            predictions[i] = np.argmax(buffer)
            
        # Final result
        return predictions
         
        
    
    # Returns the class to which the sentence belongs 
    def get_accuracy(self, outputs, labels):
        return round(np.mean(outputs == labels), 6) * 100.0
    
    
        

        
