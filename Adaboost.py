import os
import pickle

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

csv = pd.read_csv('../train.csv')
# turn categoricals to new column with 0 or 1 (except TARGET)
train = pd.get_dummies(csv, columns=['F_ACLSWKR', 'F_EDUCATION', 'F_STATUSMARIT', 'F_AMJIND', 'F_AMJOCC', 'F_RACE',
                                     'F_ORIGIN', 'F_ASEX', 'F_AWKSTAT', 'F_FILESTATUS', 'F_HHDFMX', 'F_HHDREL',
                                     'F_CONBIRTHFATH', 'F_CONBIRTHMOTH', 'F_PENATVTY', 'F_PRCITSHP'])
train_wo_target = train.drop('TARGET', 1)


def adaboost(estimator, n_estimators):
    estimator_name = type(estimator).__name__
    print("Adaboosting with estimator {}".format(estimator_name))
    classifier = AdaBoostClassifier(base_estimator=estimator, n_estimators=n_estimators)
    target = train['TARGET']

    classifier.fit(train_wo_target, target)
    with open('../{}{}'.format(estimator_name, n_estimators), 'wb') as file:
        print("Storing model of {} estimator with n_estimators={}".format(estimator_name, n))
        pickle.dump(obj=classifier, file=file)
    return classifier


if __name__ == '__main__':
    for n in [1, 2, 5, 10, 50, 100, 200]:
        # check if we have already obtained the model previously
        if not (os.path.isfile('../NoneType{}'.format(n)) and os.path.isfile('../DecisionTreeClassifier{}'.format(n))):
            ds = adaboost(None, n)  # decision stumps
            dt = adaboost(DecisionTreeClassifier(), n)  # decision trees
        else:
            with open('../NoneType{}'.format(n), 'rb') as file:
                ds = pickle.load(file)
            with open('../DecisionTreeClassifier{}'.format(n), 'rb') as file:
                dt = pickle.load(file)
        test_csv = pd.read_csv('../test.csv')
        test_wo_label = test_csv.drop('TARGET', 1)
        test = pd.get_dummies(test_wo_label)
        missing_columns = [item for item in list(train_wo_target.keys()) if item not in list(test.keys())]
        print("test dataset was missing {}; adding them with values 0".format(missing_columns))
        for mc in missing_columns:
            test[mc] = pd.Series(0, index=test.index)
        dss = ds.score(test, test_csv['TARGET'])
        print(
            "The Adaboost method using decision stumps scores a precision of {} for n_estimators={}".format(dss, n))
        dts = dt.score(test, test_csv['TARGET'])
        print(
            "The Adaboost method using decision trees scores a precision of {} for n_estimators={}".format(dts, n))
        print("Classification report for Adaboost DS n_estimators={}".format(n))
        print(classification_report(test_csv['TARGET'], ds.predict(test)))
        print("Classification report for Adaboost DT n_estimators={}".format(n))
        print(classification_report(test_csv['TARGET'], dt.predict(test)))
