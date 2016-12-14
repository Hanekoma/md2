import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#
#   **** Loading data
#

url = "http://mlr.cs.umass.edu/ml/machine-learning-databases/iris/iris.data"

headers = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
dataset = pd.read_csv(url, header=None, names=headers)

print(dataset.shape)
X = np.array(dataset[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']])
y = np.array(dataset['Class'])


#
#   **** Some printing
#

from sklearn import preprocessing
import matplotlib.pyplot as plt

le = preprocessing.LabelEncoder()
le.fit(y)
plt.scatter(X[:, 0], X[:, 1], c=le.transform(y), cmap=plt.cm.Paired)
plt.show()

from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from os import system

#
#   **** Simple cross-validation
#

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB().fit(X_train, y_train)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)     
y_pred = clf.predict(X_test)
print "Classification Report:"
print metrics.classification_report(y_test, y_pred)
print "Confusion Matrix:"
print metrics.confusion_matrix(y_test, y_pred)

# Only for Decision Trees - Print the tree 
# Graphviz should be installed 
from sklearn import tree
dotfile = open("a.dot","w")
tree.export_graphviz(clf, out_file = dotfile, feature_names = headers)
dotfile.close()
system("dot -Tpng a.dot -o a.png")


#
#   **** k-folds cross-validation
#

from sklearn.cross_validation import cross_val_score

#clf = KNeighborsClassifier(n_neighbors=3)
#clf = GaussianNB()
clf = DecisionTreeClassifier(random_state=0)
print cross_val_score(clf, X, y, cv=10)
print np.mean(cross_val_score(clf, X, y, cv=10))

y_pred = cross_validation.cross_val_predict(clf, X, y, cv=10)
print metrics.classification_report(y, y_pred)
print metrics.confusion_matrix(y, y_pred)


