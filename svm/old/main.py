import sys
import numpy as np
IN_COLAB = 'google.colab' in sys.modules

from naive_bayes_model import NaiveBayesWithSmoothing
from text_classifier import TextClassifier

train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'
submission_file = './output/submission.csv'

if IN_COLAB:
    train_set_path = 'data_train.pkl'
    test_set_path = 'data_test.pkl'
    submission_file = 'submission.csv'
else:
    train_set_path = './data/data_train.pkl'
    test_set_path = './data/data_test.pkl'
    submission_file = './output/submission.csv'
    
# Choosing classifier
classifier = TextClassifier(NaiveBayesWithSmoothing, 0.2)

# Tuning hyperparameter
classifier.fit_model(train_set_path, [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]) # TEST!!!!

# Training the model
classifier.train_model(train_set_path)

# Computing predictions 
classifier.dump_predictions(test_set_path, submission_file)

