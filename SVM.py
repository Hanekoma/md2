import pandas as pd
import numpy as np
import scipy as sp
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import svm

url = "https://doc-04-20-docs.googleusercontent.com/docs/securesc/osjkbq4hhpniq345amjh9hsp5olv3c99/181cltk6hpksqjju2ridp53bj7qa2efn/1481817600000/13264354433125755015/10308915816099072449/0BzOuT6s50TL4eHFUTFE5V1BEN1U?e=download&nonce=63ghqij6hhb7c&user=10308915816099072449&hash=ufv3h7qh066lm43act8rnp39t09kmp9v"

headers = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
dataset = pd.read_csv(url, header=None, names=headers)

print(dataset.shape)
X = np.array(dataset[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']])
y = np.array(dataset['Class'])


from sklearn import preprocessing
import matplotlib.pyplot as plt

le = preprocessing.LabelEncoder()
le.fit(y)
plt.scatter(X[:, 0], X[:, 1], c=le.transform(y), cmap=plt.cm.Paired)
plt.show()


#
#   **** Simple cross-validation
#

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)


#
#   **** Aplicamos SVM ****
#

# SVM linear svm.SVC(C, kernel)
linear_svc = svm.SVC(C=10, kernel='linear').fit(X_train, y_train)
linear_svc.fit(X_train, y_train)

# SVM polynomial svm.SVC(C, kernel, degree)
poly_2 = svm.SVC(C=10, kernel='poly', degree=2).fit(X_train, y_train)

poly_3 = svm.SVC(C=10, kernel='poly', degree=3).fit(X_train, y_train)


# SVM rbf svm.SVC(C, kernel, gamma)
rbf = svm.SVC(C=10, kernel="rbf", gamma=0.1).fit(X_train, y_train)

# Predict
linear_svc.score(X_test, y_test)
y_pred = linear_svc.predict(X_test)

print "Classification Report:"
print metrics.classification_report(y_test, y_pred)
print "Confusion Matrix:"
print metrics.confusion_matrix(y_test, y_pred)


#
#   **** k-folds cross-validation
#

#clf = KNeighborsClassifier(n_neighbors=3)
#clf = GaussianNB()
clf = svm.SVC(C=10, kernel="linear")
print cross_val_score(clf, X, y, cv=10)
print np.mean(cross_val_score(clf, X, y, cv=10))

y_pred = cross_validation.cross_val_predict(clf, X, y, cv=10)
print metrics.classification_report(y, y_pred)
print metrics.confusion_matrix(y, y_pred)
