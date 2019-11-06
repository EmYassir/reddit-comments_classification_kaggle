from classifier import preprocess_text
from classifier import create_tfidf_vector
from classifier import select_features
from classifier import split_data
from classifier import test_model

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB

# Loading data
train_set_path = './data/data_train.pkl'
test_set_path = './data/data_test.pkl'



# Experiment with countVectorizer + TfidfTransformer
print('### Begin experimentation')
# Preprocessing text
X, y = preprocess_text(train_set_path)

# Creating words vectors
X = create_tfidf_vector(X)

# Feature selection
X = select_features(15000, X, y)

# Splitting data
X_train, X_test, y_train, y_test = split_data(X, y, 0.015)


# Instantiating models
models = {}
#models['Multinomial Naive Bayes 0.01'] = MultinomialNB(alpha=0.01)
#models['Multinomial Naive Bayes 0.03'] = MultinomialNB(alpha=0.03)
models['Multinomial Naive Bayes 0.05'] = MultinomialNB(alpha=0.05)
models['Complement Naive Bayes 0.05'] = ComplementNB(alpha=0.05)
#models['Multinomial Naive Bayes 0.07'] = MultinomialNB(alpha=0.07)
#models['Multinomial Naive Bayes 0.09'] = MultinomialNB(alpha=0.09)
#models['Multinomial Naive Bayes 0.10'] = MultinomialNB(alpha=0.10)
#models['Logistic Regression lbfgs'] = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter = 1000)
#models['Logistic Regression saga'] = LogisticRegression(solver='saga', multi_class='multinomial', max_iter = 1000)
#models['Logistic Regression saga l1'] = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter = 1000)
#models['Linear Support Vector Machine l1'] = LinearSVC(penalty='l1', loss='squared_hinge', dual = False, max_iter=5000)
#models['Linear Support Vector Machine l2 sq-hinge'] = LinearSVC(penalty='l2', loss='squared_hinge', max_iter=5000)
#models['Linear Support Vector Machine l2 hinge'] = LinearSVC(penalty='l2', loss='hinge', max_iter=100000)
#models['Linear Support Vector Machine with SGD l1'] = SGDClassifier(penalty='l1', loss='squared_hinge', max_iter=5000)
#models['Linear Support Vector Machine with SGD l2 sq-hinge'] = SGDClassifier(penalty='l2', loss='squared_hinge', max_iter=5000)
#models['Linear Support Vector Machine with SGD l2 hinge'] = SGDClassifier(penalty='l2', loss='hinge', max_iter=5000)
#models['Support Vector Machine'] = SVC(kernel='poly') #Infinite loop: can't work with non normalized data
#models['Random Forest'] = RandomForestClassifier(n_estimators=200, max_depth=25)

# Testing
for key,val in models.items():
    test_model(key, val, X_train, X_test, y_train, y_test)
    
print('### End of experiment')



    