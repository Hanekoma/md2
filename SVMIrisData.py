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


def transformGetData(df):  # esta funcion debe devolver los datos en un np.array()
    # First Normalize numerical
    AAGE = normalizeAttribute(df['AAGE'])
    WKSWORKYEAR = normalizeAttribute(df['WKSWORKYEAR'])
    CHILDS = normalizeAttribute(df['CHILDS'])
    # MARSUPWT = normalizeAttribute(df['MARSUPWT']) NO FERLA SERVIR
    # then categorical
    F_ACLSWKR = np.array(pd.get_dummies(df['F_ACLSWKR'])).tolist()
    F_EDUCATION = np.array(pd.get_dummies(df['F_EDUCATION'])).tolist()
    F_STATUSMARIT = np.array(pd.get_dummies(df['F_STATUSMARIT'])).tolist()
    F_AMJIND = np.array(pd.get_dummies(df['F_AMJIND'])).tolist()
    F_AMJOCC = np.array(pd.get_dummies(df['F_AMJOCC'])).tolist()
    F_RACE = np.array(pd.get_dummies(df['F_RACE'])).tolist()
    F_ORIGIN = np.array(pd.get_dummies(df['F_ORIGIN'])).tolist()
    F_ASEX = np.array(pd.get_dummies(df['F_ASEX'])).tolist()
    F_AWKSTAT = np.array(pd.get_dummies(df['F_AWKSTAT'])).tolist()
    F_FILESTATUS = np.array(pd.get_dummies(df['F_FILESTATUS'])).tolist()
    F_HHDFMX = np.array(pd.get_dummies(df['F_HHDFMX'])).tolist()
    F_HHDREL = np.array(pd.get_dummies(df['F_HHDREL'])).tolist()
    F_CONBIRTHFATH = np.array(pd.get_dummies(df['F_CONBIRTHFATH'])).tolist()
    F_CONBIRTHMOTH = np.array(pd.get_dummies(df['F_CONBIRTHMOTH'])).tolist()
    F_PENATVTY = np.array(pd.get_dummies(df['F_PENATVTY'])).tolist()
    F_PRCITSHP = np.array(pd.get_dummies(df['F_PRCITSHP'])).tolist()
    res = []
    for i in range(len(AAGE)):
        res.append(
            F_ACLSWKR[i] + F_EDUCATION[i] + F_STATUSMARIT[i] + F_AMJIND[i] + F_AMJOCC[i] + F_RACE[i] + F_ORIGIN[i] +
            F_ASEX[i] + F_AWKSTAT[i] + F_FILESTATUS[i] + F_HHDFMX[i] + F_HHDREL[i] + F_CONBIRTHFATH[i] + F_CONBIRTHMOTH[
                i] + F_PENATVTY[i] + F_PRCITSHP[i] + [AAGE[i], WKSWORKYEAR[i], CHILDS[i]])
    return res


def mainfunction():
    # Loading the Digits dataset
    url = "C:\Users\sergi\Desktop\WorkSpace\md2\Census.csv"
    df = pd.read_csv(url)
    X = np.array(transformGetData(df))
    y = np.array(df['TARGET'])

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

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

        pca = PCA().fit(X_train)

        PcaX = pca.transform(X_train)

        # Generate grid along first two principal components
        multiples = np.arange(-2, 2, 0.1)
        # steps along first component
        for j in range(5):
            for m in range(5):
                first = multiples[:, np.newaxis] * pca.components_[j, :]
                # steps along second component
                second = multiples[:, np.newaxis] * pca.components_[m, :]
                # combine
                grid = first[np.newaxis, :, :] + second[:, np.newaxis, :]
                flat_grid = grid.reshape(-1, X.shape[1])

                # title for the plots
                titles = ['SVC with rbf kernel',
                          'SVC (linear kernel)\n with Fourier rbf feature map\n'
                          'n_components=100',
                          'SVC (linear kernel)\n with Nystroem rbf feature map\n'
                          'n_components=100']

                #plt.tight_layout()
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 3, 1)
                sv = clf.best_estimator_.fit(X_train, y_train)
                Z = sv.predict(flat_grid)
                Z2 = np.array(pd.get_dummies(Z)).astype(float)
                # Put the result into a color plot
                Z = Z2.reshape(grid.shape[:-1])
                plt.contourf(multiples, multiples, Z, cmap=plt.cm.Paired)
                plt.axis('off')
                colors = np.array(pd.get_dummies(y_train)).astype(float)
                le = LabelEncoder()
                col2 = le.fit(y_train).transform(y_train)
                # Plot also the training points
                plt.scatter(PcaX[:, j], PcaX[:, m], c=col2, cmap=plt.cm.Paired)
                plt.title("PCA:" + str(j) + "and: "+ str(m))
                plt.show()




if __name__ == '__main__':
    mainfunction()
