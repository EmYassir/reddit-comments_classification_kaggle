import numpy as np
import pickle as pkl


class NaiveBayesWithSmoothing:
    
    def __init__(self, *hyperparameters):
        argCount = len(hyperparameters)
        if argCount == 0:
            self._alpha = 0.0
        else:
            self._alpha = float((hyperparameters[0])[0])
        self._X = None
        self._y = None
        self._Xtrain = None
        self._ytrain = None
        self._Xval = None
        self._yval = None
        self._classes = None
        self._wordProbVec = None
        self._classProbVec = None
        
    def load(self, inFilePath = 'model.npy'):
        pass
    
    def dump(self, outFilePath = 'model.npy'):
        pass
    
    def split_data(self, X):
        print("X.shape:")
        print(X.shape)
        y = X[:, -1]
        d = X.shape[1] - 1
        n = X.shape[0]
        n_train = 0.85 * n
        n_val = n - n_train
        self._Xtrain = np.empty((0,d))
        self._Xval = np.empty((0,d))
        self._ytrain = np.empty((0,))
        self._yval = np.empty((0,))
        n_class = self._classes.shape[0]
        for i in range(n_class):
            # getting samples of class i
            c_idx = np.where(y == i)[0]
            portion = X[c_idx]
            
            # Filling Training set
            n_per_train = int(n_train/n_class)
            self._Xtrain = np.vstack((self._Xtrain, portion[0:n_per_train,:-1]))
            self._ytrain= np.hstack((self._ytrain, portion[0:n_per_train, -1]))
            
            # Filling Validation set
            n_per_val = int(n_val/n_class)
            s_val = n_per_train + 1
            e_val = n_per_train + n_per_val      
            self._Xval = np.vstack((self._Xval, portion[s_val:e_val,:-1]))
            self._yval = np.hstack((self._yval, portion[s_val:e_val,-1]))
        
        for i in range(n_class):
            percentage = np.mean(self._ytrain == i)
            #print('percentage of class %d in training set is %.2f %%' %(i, percentage*100.0))
            
            percentage = np.mean(self._yval == i)
            #print('percentage of class %d in validation set is %.2f %%' %(i, percentage*100.0))
        

    def get_classes(self):
        return self._classes
    
    def log_prob(self, p):
        if p != 0.0:
            return np.log(p)
        return 0.0

    def train(self, data, labels, classes):
        print('Training the classifier...')
        self._X = data
        
        self._y = labels
        n, d = self._X.shape
        # Getting classes
        self._classes = classes
        # Computing classes probabilities
        self._classProbVec = np.zeros(self._classes.shape)
        for i, c in enumerate(self._classes):
            c_idx = np.where(self._y == i)[0]
            self._classProbVec[i] = self.log_prob(c_idx.shape[0]) - self.log_prob(n)
            
        # Computing words likelihoods
        self._wordProbVec = np.zeros((d, self._classes.shape[0]))
        for i, c in enumerate(self._classes):
            c_idx = np.where(self._y == i)[0]
            log_denominator = self.log_prob(np.sum(self._X[c_idx]) + self._alpha * d)
            for v in range(d):
                numerator = np.sum(self._X[c_idx, v], axis=0) + self._alpha
                self._wordProbVec[v, i] = self.log_prob(numerator) - log_denominator

    # Returns the class to which the sentence belongs 
    def predict(self, X):
        print('Predicting new comments...')
        n = X.shape[0]
        predictions = np.zeros((n, self._classes.shape[0]))
        for i, ex in enumerate(X):
            for k, c in enumerate(self._classes):
                p = 0.0
                idx_list = np.where(ex != 0)[0]  # TODO handle the case where no words in the voc are present
                for j in idx_list:
                    p += self._wordProbVec[j, k] 
                predictions[i, k] = p + self._classProbVec[k]
        return np.argmax(predictions, axis=1)
    
    # Returns the class to which the sentence belongs 
    def accuracy(self, outputs, labels):
        return round(np.mean(outputs == labels), 2) * 100.0
        
        
    # Training + Validation
    def fit(self, X, classes):
        self._classes = classes
        # Splitting between Training/Validation sets
        self.split_data(X)
        print('Xtrain (%d,%d):' %(self._Xtrain.shape[0],self._Xtrain.shape[1]))
        #print(self._Xtrain)
        #print('Ytrain:')
        #print(self._ytrain)
        print('Xval(%d,%d):' %(self._Xval.shape[0],self._Xval.shape[1]))
        #print(self._Xval)
        #print('Yval')
        print(self._yval)
        # Defining the range of hyperparameter alpha
        hyper_params = np.linspace(0.0,1.0,20)
        #hyper_params = [0.5]
        alpha_star = 0.0 
        # Best score seen so far
        best_score = 0.0
        print('Fitting model...')
        
        # Training with different alphas
        for alpha in hyper_params:
            self._alpha = alpha
            self.train(self._Xtrain, self._ytrain, classes)
            outputs = self.predict(self._Xval)
            new_score = self.accuracy(outputs, self._yval)
            #print('outputs shape:')
            #print(outputs.shape)
            #print('values.shape')
            #print(self._yval.shape)
            print('Accuracy of %.2f %% for alpha == %.2f' %(new_score, alpha))
            if (new_score > best_score):
                best_score = new_score
                alpha_star = alpha
        self._alpha = alpha_star
        print('Got accuracy of %.2f %% for alpha* == %.2f' %(best_score,alpha_star))
        
