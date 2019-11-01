import csv
import numpy as np
import pickle as pkl
import random as rand
from nlp_util import NlpUtil
from dictionary import Dictionary

class TextClassifier:
    # init
    def __init__(self, model, *args):
        self._model = model(args)
        self._nlpUtil = NlpUtil()
        self._dico = None
        self._labels = None
        self._labels_Ids = {}
        
     # Initializes labels structure 
    def _init_numerical_labels(self, y):
        # Initializes list of available labels
        self._labels = np.unique(y)
        self._labels_Ids = {}
        # Assigning fixed ids to labels
        for i in range(self._labels.shape[0]):
            self._labels_Ids[self._labels[i]] = i
        
    # Shuffles rows 
    def _convert_to_numerical_labels(self, y):
        lbl = []
        for l in y:
            lbl.append(self._labels_Ids[l])
        return np.array(lbl).astype(int)
        
    # Shuffles rows 
    def _shuffle_rows(self, X, y):
        if (X.shape[0] != y.shape[0]):
            print('Not possible to shuffle arrays with different shapes together')
            return X, y
        # Shuffling indexes
        indexes = list(range(X.shape[0]))
        rand.shuffle(indexes)
        # Shuffled arrays
        return X[indexes],y[indexes]
    
    # Splits data into traning and validation sets
    def _split_data(self, X, y, percent = 0.9):
        if (X.shape[0] != y.shape[0]):
            print('Not possible to shuffle arrays with different shapes together')
            return None, None, None, None
        n = X.shape[0]
        c = self._labels.shape[0]
        # Adjusting p
        p = percent
        if p < 0.65 or p > 0.95:
            p = 0.90
        
        # Adjusting numbers
        n_train = (n * p) - ((n * p) % c)    
        n_val = (n - n_train) - ((n - n_train) % c)
        
        k_train = int(n_train / c)
        k_val = int(n_val / c)
        
        Xtrain = np.empty((0,))
        yval = np.empty((0,))
        Xval = np.empty((0,))
        ytrain = np.empty((0,))

  
        for i in range(c):
            # getting samples of class i
            c_idx = np.where(y == i)[0]
            Xs, ys = self._shuffle_rows(X[c_idx], y[c_idx])
            
            # Filling Training set
            Xtrain = np.hstack((Xtrain, Xs[0:k_train]))
            ytrain= np.hstack((ytrain, ys[0:k_train]))
            
            # Filling Validation set
            s_val = k_train
            e_val = k_train + k_val  
            Xval = np.hstack((Xval, Xs[s_val:e_val]))
            yval = np.hstack((yval, ys[s_val:e_val]))
        
        # Validation
        '''
        for i in range(c):
            percentage = np.mean(ytrain == i)
            print('percentage of class %d in training set is %.6f %%' %(i, percentage*100.0))           
            percentage = np.mean(yval == i)
            print('percentage of class %d in validation set is %.6f %%' %(i, percentage*100.0))
        '''
        return  Xtrain, ytrain.astype(int), Xval, yval.astype(int)
        
    def _load_data(self, data_path):
        return np.load(data_path, allow_pickle=True)
    
    
    def train_model(self, train_set_path):
        print('Loading data...')
        comms, lbls = self._load_data(train_set_path)
        comments = np.array(comms) 
        labels = np.array(lbls) 
        
        print('Preprocessing data...')
        # List of tokenized sentences
        X = self._nlpUtil.get_preprocessed_data(comments)
        # labels
        self._init_numerical_labels(labels)
        y = self._convert_to_numerical_labels(labels)
        
        # Dictionary
        print('Updating dictionary...')
        self._dico = Dictionary(self._convert_to_numerical_labels(self._labels))
        #self._dico.update_mask(['like','one'])
        for i, lwords in enumerate(X):
            self._dico.update(lwords, int(y[i]))
        
        # Training the model
        print('Training the model...')
        self._model.train(self._dico, X, y)
    
     # Getting predictions
    def _get_predictions(self, test_set_path):
        print('Predicting new data...')
        data = np.array(self._load_data(test_set_path))
        self._nlpUtil.get_preprocessed_data(data)
        return self._model.predict(data)
    
    
     # Training + Validation
    def fit_model(self, train_set_path, hypers = []):
        values = None
        if len(hypers) > 0:
            values = np.array(hypers)
        else:
            values = np.linspace(0.0,1.0,20)
            
        print('Fitting the model...')
        print('-> Loading raw data...')
        comms, lbls = self._load_data(train_set_path)
        comments = np.array(comms) 
        labels = np.array(lbls) 
        # labels
        print('-> Processing labels...')
        self._init_numerical_labels(labels)
        y = self._convert_to_numerical_labels(labels)
    
        print('-> Preprocessing raw data...')
        X = self._nlpUtil.get_preprocessed_data(comments)
        
        # Splitting data
        print('-> Splitting data...')
        Xtrain, ytrain, Xval, yval = self._split_data(X, y, 0.98)
        print(Xtrain.shape[0])
        print(Xval.shape[0])
        # Dictionary
        print('-> Updating dictionary...')
        self._dico = Dictionary(self._convert_to_numerical_labels(self._labels))
        #self._dico.update_mask(['like','one'])
        for i, lwords in enumerate(Xtrain):
            self._dico.update(lwords, int(ytrain[i]))
        # For debug
        print('Dictionary has total of %d words' %self._dico.get_global_count())
        self._dico.dump_dictionary('all.txt')
        self._dico.dump_dictionary_per_classes(20, 'classes.txt')
            
        alpha_star = 0.0 
        best_score = 0.0
        print('-> Iterating')
        for alpha in values:
            print('  -> 1. Train')
            self._model.set_alpha(alpha)
            self._model.train(self._dico, Xtrain, ytrain)
            print('  -> 2. Predict')
            outputs = self._model.predict(Xval)
            print('  -> 3. Validate:')
            new_score = self._model.get_accuracy(outputs, yval)
            print('        => Accuracy of %.2f %% for alpha == %.2f' %(new_score, alpha))
            if (new_score > best_score):
                best_score = new_score
                alpha_star = alpha
        
        # Updating the model with alpha_star
        self._model.set_alpha(alpha_star)
        
        

    def dump_predictions(self, test_set_path, filepath="submission.csv"):
        predictions = self._get_predictions(test_set_path)
        print(predictions)
        with open(filepath, mode='w', newline='') as submission_file:
            writer = csv.writer(submission_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            writer.writerow(["Id", "Category"])
            for i, prediction in enumerate(predictions):
                writer.writerow([i, self._labels[int(prediction)]])
                

        