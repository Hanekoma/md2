from __future__ import print_function
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def normalizeAttribute(attr1):
    a = np.array(attr1)
    # PROBLEMA NO PODEM LLEGIR STRING
    a_norm = normalize(a.astype(np.float), norm='l2')
    # al fer np.array et genera una matriu.. encara que tingui dimensio 1
    return (a_norm.tolist())[0]



def mainfunction():
    # Loading the Digits dataset
    url = "S:\UNI\WorkSpace\md2\Census.csv"
    df = pd.read_csv(url)
    cleaned = pd.get_dummies(df,
                             columns=['F_ACLSWKR', 'F_EDUCATION', 'F_STATUSMARIT', 'F_AMJIND', 'F_AMJOCC', 'F_RACE',
                                      'F_ORIGIN', 'F_ASEX', 'F_AWKSTAT', 'F_FILESTATUS', 'F_HHDFMX', 'F_HHDREL',
                                      'F_CONBIRTHFATH', 'F_CONBIRTHMOTH', 'F_PENATVTY', 'F_PRCITSHP'])

    cols_to_norm = ['AAGE', 'WKSWORKYEAR', 'CHILDS']
    cleaned[cols_to_norm] = cleaned[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    cleaned = cleaned.drop('MARSUPWT',1)
    cleaned_wo_target = cleaned.drop('TARGET', 1)
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_wo_target, cleaned['TARGET'], test_size=0.25)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score, n_jobs=8)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()



if __name__ == '__main__':
    mainfunction()
