import csv
import numpy as np
import pickle as pkl
from text_processing_util import TextProcessingUtil



class TextClassifier:
    
    def __init__(self, model, *args):
        self.model = model(args)
        self._util = TextProcessingUtil()
        
    def train_model(self, train_set_path):
        comments, labels = np.load(train_set_path, allow_pickle=True)
        data = self._util.get_bow_matrix(comments, labels)
        X = data[:,0:-1]
        y = data[:,-1]
        self.model.train(X, y, self._util.get_labels())
        
    def fit_model(self, train_set_path):
        comments, labels = np.load(train_set_path, allow_pickle=True)
        data = self._util.get_bow_matrix(comments, labels)
        self.model.fit(data, self._util.get_labels())
        
    def get_predictions(self, test_set_path):
        comments = np.load(test_set_path, allow_pickle=True)
        data = self._util.get_bow_matrix(comments)
        return self.model.predict(data)
    
    def dump_predictions(self, test_set_path, filepath="submission.csv"):
        predictions = self.get_predictions(test_set_path)
        with open(filepath, mode='w', newline='') as submission_file:
            classes = self.model.get_classes()
            writer = csv.writer(submission_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            writer.writerow(["Id", "Category"])
            for i, prediction in enumerate(predictions):
                writer.writerow([i, classes[prediction]])

    