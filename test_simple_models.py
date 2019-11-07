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

# Loading data
train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'



# Experiment with countVectorizer + TfidfTransformer
print('### Begin experimentation')
class_tool =  Classifier_Util()

# setting models to process features
class_tool.set_word_vectorizer(TfidfVectorizer())
class_tool.set_feature_selector(SelectKBest(chi2, 20000))

print('Preprocessing raw data...')
# Preprocessing text
#X, y = class_tool.preprocess_text_with_removal(train_set_path)
X, y = class_tool.preprocess_text(train_set_path)


# Splitting data
print('Splitting data...')
X_train, X_test, y_train, y_test = class_tool.split_data(X, y, 0.015)

# Creating words vectors
print('Creating word vectors...')
class_tool.fit_words_vectorizer(X_train, y_train)
X_train = class_tool.get_words_vector(X_train)
X_test = class_tool.get_words_vector(X_test)

# Feature selection
class_tool.fit_feature_selector(X_train, y_train)
X_train = class_tool.get_selected_features(X_train)
X_test = class_tool.get_selected_features(X_test)



# Instantiating models
models = {}
#models['Multinomial Naive Bayes 0.01'] = MultinomialNB(alpha=0.01)
#models['Multinomial Naive Bayes 0.03'] = MultinomialNB(alpha=0.03)
#models['Multinomial Naive Bayes 0.05'] = MultinomialNB(alpha=0.05)
models['Multinomial Naive Bayes 0.10'] = MultinomialNB(alpha=0.10)
#models['Gradient Boosting Classifier'] = GradientBoostingClassifier(n_estimators=100, max_depth=3)
#models['Gradient Bagging Classifier'] = BaggingClassifier(n_estimators=200)
#models['Random Forest'] = RandomForestClassifier(n_estimators=200, max_depth=5)
#models['Logistic Regression lbfgs'] = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter = 1000)
#models['Logistic Regression saga'] = LogisticRegression(solver='saga', multi_class='multinomial', max_iter = 1000)
#models['Logistic Regression saga l1'] = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter = 1000)
#models['Linear Support Vector Machine l1'] = LinearSVC(penalty='l1', loss='squared_hinge', dual = False, max_iter=5000)
models['Linear Support Vector Machine l2 sq-hinge'] = LinearSVC(penalty='l2', loss='squared_hinge', max_iter=5000)
#models['Linear Support Vector Machine l2 hinge'] = LinearSVC(penalty='l2', loss='hinge', max_iter=100000)
#models['Linear Support Vector Machine with SGD l1'] = SGDClassifier(penalty='l1', loss='squared_hinge', max_iter=5000)
#models['Linear Support Vector Machine with SGD l2 sq-hinge'] = SGDClassifier(penalty='l2', loss='squared_hinge', max_iter=5000)
models['Linear Support Vector Machine with SGD l2 hinge'] = SGDClassifier(penalty='l2', loss='hinge', max_iter=5000)
#models['Support Vector Machine'] = SVC(kernel='poly') #Infinite loop: can't work with non normalized data
#models['Random Forest'] = RandomForestClassifier(n_estimators=200, max_depth=5)

# Testing
'''
for key,val in models.items():
   class_tool.test_model(key, val, X_train, X_test, y_train, y_test)
'''
class_tool.set_voting_classifier(models, VotingClassifier)
class_tool.test_voting_models(X_train, X_test, y_train, y_test)
print('### End of experiment')



    