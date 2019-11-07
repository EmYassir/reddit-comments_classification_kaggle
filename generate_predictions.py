from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from classifier import Classifier_Util
import sys

IN_COLAB = 'google.colab' in sys.modules

# Loading data
train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'
output_file = './output/submission.csv'

if IN_COLAB:
    train_set_path = 'data_train.pkl'
    test_set_path = 'data_test.pkl'
    




# Experiment with countVectorizer + TfidfTransformer
print('### Begin ')
class_tool =  Classifier_Util()

# setting models to process features
class_tool.set_word_vectorizer(TfidfVectorizer())
class_tool.set_feature_selector(SelectKBest(chi2, 20000))

print('Preprocessing raw data...')
# Preprocessing text
X_train, y_train = class_tool.preprocess_text(train_set_path)
X_test, y_test = class_tool.preprocess_text(test_set_path, test = True)


# Creating words vectors
print('Creating word vectors...')
class_tool.fit_words_vectorizer(X, y)
X_train = class_tool.get_words_vector(X_train)
X_test = class_tool.get_words_vector(X_test)

# Feature selection
print('Selecting features...')
class_tool.fit_feature_selector(X_train, y_train)
X_train = class_tool.get_selected_features(X_train)
X_test = class_tool.get_selected_features(X_test)

# Setting voting classifier
models = {}
models['Multinomial Naive Bayes 0.10'] = MultinomialNB(alpha=0.10)
models['Linear Support Vector Machine l2 sq-hinge'] = LinearSVC(penalty='l2', loss='squared_hinge', max_iter=5000)
models['Linear Support Vector Machine with SGD l2 hinge'] = SGDClassifier(penalty='l2', loss='hinge', max_iter=5000)
class_tool.set_voting_classifier(models, VotingClassifier)

class_tool.train_predict_dump(X_train, y_train, X_test)
print('### End of program')



    