#Libraries imports
import csv
import random
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import sklearn.neighbors as nb
import sklearn.cross_validation as cv
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#myImports

#Code
def normalizeAttribute(attr1):
	a = np.array(attr1)
	#PROBLEMA NO PODEM LLEGIR STRING
	a_norm = normalize(a.astype(np.float), norm='l2')
	#al fer np.array et genera una matriu.. encara que tingui dimensio 1
	return (a_norm.tolist())[0]

def transformGetData(df): #esta funcion debe devolver los datos en un np.array()
	#First Normalize numerical
	AAGE = normalizeAttribute(df['AAGE'])	
	WKSWORKYEAR = normalizeAttribute(df['WKSWORKYEAR'])
	CHILDS = normalizeAttribute(df['CHILDS'])
	#MARSUPWT = normalizeAttribute(df['MARSUPWT']) NO FERLA SERVIR
	#then categorical
	F_ACLSWKR =  np.array(pd.get_dummies(df['F_ACLSWKR'])).tolist()
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
		res.append(F_ACLSWKR[i]+F_EDUCATION[i]+F_STATUSMARIT[i]+F_AMJIND[i]+F_AMJOCC[i]+F_RACE[i]+F_ORIGIN[i]+F_ASEX[i]+F_AWKSTAT[i]+F_FILESTATUS[i]+F_HHDFMX[i]+F_HHDREL[i]+F_CONBIRTHFATH[i]+F_CONBIRTHMOTH[i]+F_PENATVTY[i]+F_PRCITSHP[i]+[AAGE[i],WKSWORKYEAR[i],CHILDS[i]])
	return res

def loadCross(filename,split):
	#LoadDataSet
	#headers = ["index","AAGE","F_ACLSWKR","F_EDUCATION","F_STATUSMARIT","F_AMJIND","F_AMJOCC","F_RACE","F_ORIGIN","F_ASEX","F_AWKSTAT","F_FILESTATUS","F_HHDFMX","F_HHDREL","MARSUPWT","CHILDS","F_CONBIRTHFATH","F_CONBIRTHMOTH","F_PENATVTY","F_PRCITSHP","WKSWORKYEAR","TARGET"]
	df = pd.read_csv(filename)
	X_data=transformGetData(df)
	y_labels = np.array(df['TARGET'])
	#Cross-validation split
	(dTrain,dTest,labTrain,labTest) = cv.train_test_split(X_data, y_labels, test_size=split)
	return (dTrain,dTest,labTrain,labTest)

def doKNN(dataSetName):
	#Do Cross
	(dTrain,dTest,labTrain,labTest) = loadCross(dataSetName,.25)
	#Do KNN
	print("---We have arrived to KNN classifier----")
	scoreK = [] 
	for neighbor in range(1,15,2):
		knn = nb.KNeighborsClassifier(n_neighbors=neighbor	,weights='uniform',algorithm='auto',leaf_size=30, metric='minkowski',metric_params=None,p=2, n_jobs=4)
		knn.fit(dTrain, labTrain)
		scoreK.append(knn.score(dTest, labTest))
	plt.plot(range(1,15,2),scoreK,'b')
	#That part is too slow ..
	'''
	for we in ['uniform', 'distance']:
		knn = nb.KNeighborsClassifier(n_neighbors=5	,weights=we,algorithm='auto',leaf_size=30, metric='minkowski',metric_params=None,p=2, n_jobs=4)
		knn.fit(dTrain, labTrain);
		#Print it
		print('Score using {} : {}'.format(we,knn.score(dTest, labTest)))
		cv_scores = cross_val_score(knn,X=dTest,y=labTest,cv=10, scoring='f1_macro')
		#f1_macro = Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
		#F1 = 2 * (precision * recall) / (precision + recall)
		# imprimim els resultats de cada folder
		print(cv_scores)
		# obtenim la mitjana de les 10 execucions
		print(np.mean(cv_scores))
	'''
#MAIN
doKNN('5000Census.csv')

