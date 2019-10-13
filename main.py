import numpy as np
import csv
from naive_bayes_model import NaiveBayesWithSmoothing


def get_trained_model(train_set_path):
    comments, labels = np.load(train_set_path, allow_pickle=True)
    model = NaiveBayesWithSmoothing(0.5)
    model.train(comments, labels)
    return model


def get_predictions(model, test_set_path):
    comments = np.load(test_set_path, allow_pickle=True)
    return model.predict(comments)


def dump_predictions(model, predictions, output_path=""):
    filepath = str(output_path) + '/submission.csv'
    with open(filepath, mode='w', newline='') as submission_file:
        classes = model.get_classes()
        writer = csv.writer(submission_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
        writer.writerow(["Id", "Category"])
        for i, prediction in enumerate(predictions):
            writer.writerow([i, classes[prediction]])


nb_model = get_trained_model(str('./data/data_train.pkl'))
preds = get_predictions(nb_model, str('./data/data_test.pkl'))
dump_predictions(nb_model, preds, str('./output'))

