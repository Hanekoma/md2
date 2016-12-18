import pandas as pd
import pickle
import os

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

csv = pd.read_csv('../train.csv')
# turn categoricals to new column with 0 or 1 (except TARGET)
train = pd.get_dummies(csv, columns=['F_ACLSWKR', 'F_EDUCATION', 'F_STATUSMARIT', 'F_AMJIND', 'F_AMJOCC', 'F_RACE',
                                     'F_ORIGIN', 'F_ASEX', 'F_AWKSTAT', 'F_FILESTATUS', 'F_HHDFMX', 'F_HHDREL',
                                     'F_CONBIRTHFATH', 'F_CONBIRTHMOTH', 'F_PENATVTY', 'F_PRCITSHP'])
train_wo_target = train.drop('TARGET', 1)


def adaboost(estimator):
    estimator_name = type(estimator).__name__
    print("Adaboosting with estimator {}".format(estimator_name))
    classifier = AdaBoostClassifier(base_estimator=estimator, n_estimators=200)
    target = train['TARGET']

    classifier.fit(train_wo_target, target)
    with open('../{}'.format(estimator_name), 'wb') as file:
        print("Storing model of {} estimator".format(estimator_name))
        pickle.dump(obj=classifier, file=file)
    return classifier


if __name__ == '__main__':
    # check if we have already obtained the model previously
    if not (os.path.isfile('../NoneType') and os.path.isfile('../DecisionTreeClassifier')):
        ds = adaboost(None)  # decision stumps
        dt = adaboost(DecisionTreeClassifier())  # decision trees
    else:
        with open('../NoneType', 'rb') as file:
            ds = pickle.load(file)
        with open('../DecisionTreeClassifier', 'rb') as file:
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
        "The Adaboost method using decision stumps scores a precision of {}".format(dss))
    dts = dt.score(test, test_csv['TARGET'])
    print(
        "The Adaboost method using decision trees scores a precision of {}".format(dts))
