from naive_bayes_model import NaiveBayesWithSmoothing
from text_classifier import TextClassifier

train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'
submission_file = './output/submission.csv'

classifier = TextClassifier(NaiveBayesWithSmoothing, 0.05)
# classifier.fit_model(train_set_path)
classifier.train_model(train_set_path)
classifier.dump_predictions(test_set_path, submission_file)

