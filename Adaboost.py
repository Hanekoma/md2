import pandas as pd
import pickle
import os

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

csv = pd.read_csv('../censusbuenodeverdad.csv')
# turn categoricals to new column with 0 or 1 (except TARGET)
cleaned = pd.get_dummies(csv, columns=['F_ACLSWKR', 'F_EDUCATION', 'F_STATUSMARIT', 'F_AMJIND', 'F_AMJOCC', 'F_RACE',
                                       'F_ORIGIN', 'F_ASEX', 'F_AWKSTAT', 'F_FILESTATUS', 'F_HHDFMX', 'F_HHDREL',
                                       'F_CONBIRTHFATH', 'F_CONBIRTHMOTH', 'F_PENATVTY', 'F_PRCITSHP'])
cleaned_wo_target = cleaned.drop('TARGET', 1)
x_train, x_test, y_train, y_test = train_test_split(cleaned_wo_target, cleaned['TARGET'], test_size=0.25)


def adaboost(estimator):
    estimator_name = type(estimator).__name__
    print("Adaboosting with estimator {}".format(estimator_name))
    classifier = AdaBoostClassifier(base_estimator=estimator, n_estimators=200)
    target = y_train

    classifier.fit(x_train, target)
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
    print("The Adaboost method using decision stumps scores a precision of {}".format(ds.score(x_test, y_test)))
    print("The Adaboost method using decision trees scores a precision of {}".format(dt.score(x_test, y_test)))
