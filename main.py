import sys
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from preprocessing import *
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics, tree, ensemble, neighbors, linear_model
import pickle
import math
import json
import itertools


class Predict:

    def __init__(self, filenameData, filenameTestdata):
        self.data = pd.read_csv(filenameData)
        self.data_test = pd.read_csv(filenameTestdata)
        self.normalizedFeatures, self.target = self.preprocessing(self.data)
        self.normalizedTestFeatures = self.test_preprocessing(self.data_test)

    def preprocessing(self, data):
        # Class specific preprocessing
        data1 = data.loc[data['Target (Col 107)'] == 1]
        data0 = data.loc[data['Target (Col 107)'] == 0]

        nominalData0 = data0.iloc[:, 103:]
        numericalData0 = data0.iloc[:, :103]
        numericalData0 = numericalData0.interpolate(
            axis=0, limit_direction='both', method='linear',)
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        nominalData0 = pd.DataFrame(imputer.fit_transform(
            nominalData0), index=nominalData0.index, columns=nominalData0.columns)
        data0 = pd.concat([numericalData0, nominalData0], axis=1, join="inner")

        numericalData1 = data1.iloc[:, :103]
        nominalData1 = data1.iloc[:, 103:]
        numericalData1 = numericalData1.interpolate(
            axis=0, limit_direction='both', method='linear',)
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        nominalData1 = pd.DataFrame(imputer.fit_transform(
            nominalData1), index=nominalData1.index, columns=nominalData1.columns)
        data1 = pd.concat([numericalData1, nominalData1], axis=1, join="inner")

        data = pd.concat([data0, data1], axis=0, join="inner")

        features = data.iloc[:, :106]
        target = data.iloc[:, 106]
        normalScaler = MinMaxScaler()
        normalizedFeatures = pd.DataFrame(
            normalScaler.fit_transform(features), columns=features.columns)
        return normalizedFeatures, target

    def test_preprocessing(self, data):
        normalScaler = MinMaxScaler()

        return pd.DataFrame(
            normalScaler.fit_transform(data), columns=data.columns)

    def predict(self):
        normalizedFeatures = self.normalizedFeatures
        target = self.target

        parameters = {'criterion': ['entropy'],
                      'max_depth': range(2, 5), 'max_leaf_nodes': range(4, 5)}
        clf = GridSearchCV(DecisionTreeClassifier(random_state=14), parameters, verbose=0,
                           cv=10,  n_jobs=8, refit="f1", scoring=['accuracy', 'f1']).fit(X=normalizedFeatures, y=target)
        i = clf.best_index_
        f1 = round(clf.cv_results_['mean_test_f1'][i], 3)
        accuracy = round(clf.cv_results_['mean_test_accuracy'][i], 3)
        entropyDT = clf.best_estimator_
        resultDTE = (clf.best_score_)
        print("Decision Tree Entropy")
        print("Acc and f1: "+str((accuracy, f1)))
        print("Parameters: " + str(clf.best_params_) + '\n')

        parameters = {'criterion': ['gini'],
                      'max_depth': range(2, 3), 'max_leaf_nodes': range(3, 4)}
        clf = GridSearchCV(DecisionTreeClassifier(random_state=38), parameters, verbose=0,
                           cv=10,  n_jobs=8, refit="f1", scoring=['accuracy', 'f1']).fit(X=normalizedFeatures, y=target)
        i = clf.best_index_
        f1 = round(clf.cv_results_['mean_test_f1'][i], 3)
        accuracy = round(clf.cv_results_['mean_test_accuracy'][i], 3)
        giniDT = clf.best_estimator_
        resultDTG = (clf.best_score_)
        print("Decision Tree Gini")
        print("Acc and f1: "+str((accuracy, f1)))
        print("Parameters: " + str(clf.best_params_) + '\n')

        parameters = {'n_estimators': range(150, 151), 'criterion': ['gini', ],
                      'max_depth': range(20, 21), }
        clf = GridSearchCV(RandomForestClassifier(random_state=14), parameters, verbose=0,
                           cv=10,  n_jobs=8, refit="f1", scoring=['accuracy', 'f1']).fit(X=normalizedFeatures, y=target)
        i = clf.best_index_
        f1 = round(clf.cv_results_['mean_test_f1'][i], 3)
        accuracy = round(clf.cv_results_['mean_test_accuracy'][i], 3)
        forest = clf.best_estimator_
        resultFOR = (clf.best_score_)
        print("Random forest")
        print("Acc and f1: "+str((accuracy, f1)))
        print("Parameters: " + str(clf.best_params_) + '\n')

        parameters = {'n_neighbors': range(9, 10), 'weights': ['uniform'],
                      'p': [2]}
        neigh = GridSearchCV(KNeighborsClassifier(), parameters, verbose=0,
                             cv=10,  n_jobs=8, refit="f1", scoring=['accuracy', 'f1']).fit(X=normalizedFeatures, y=target)
        i = neigh.best_index_
        f1 = round(neigh.cv_results_['mean_test_f1'][i], 3)
        accuracy = round(neigh.cv_results_['mean_test_accuracy'][i], 3)
        knn = neigh.best_estimator_
        resultKNN = (neigh.best_score_)
        print("K-nearest neighbour")
        print("Acc and f1: "+str((accuracy, f1)))
        print("Parameters: " + str(neigh.best_params_) + '\n')

        parameters = {}
        gnb = GridSearchCV(GaussianNB(), parameters, verbose=0,
                           cv=10,  n_jobs=8, refit="f1", scoring=['accuracy', 'f1']).fit(X=normalizedFeatures, y=target)
        i = gnb.best_index_
        gnbBest = gnb.best_estimator_
        resultGNB = (gnb.best_score_)
        f1 = round(gnb.cv_results_['mean_test_f1'][i], 3)
        accuracy = round(gnb.cv_results_['mean_test_accuracy'][i], 3)
        print("Naive Bayes")
        print("Acc and f1: "+str((accuracy, f1)))
        print("Parameters: " + str(gnb.best_params_) + '\n')

        estimator = [('DecisionTree Gini', DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3)), ('Random Forest Entropy',
                                                                                                    RandomForestClassifier(max_depth=20, n_estimators=150)), ('knn', KNeighborsClassifier(n_neighbors=9))]
        parameters = {'weights': [[1, 2, 2]], 'voting': ['hard'], }
        votingEnsemble2 = GridSearchCV(VotingClassifier(
            estimator), parameters,  verbose=1, cv=10, n_jobs=8, refit="f1", scoring=['accuracy', 'f1']).fit(X=normalizedFeatures, y=target)
        i = votingEnsemble2.best_index_
        f1 = round(votingEnsemble2.cv_results_['mean_test_f1'][i], 3)
        accuracy = round(votingEnsemble2.cv_results_[
            'mean_test_accuracy'][i], 3)
        ensemble2 = votingEnsemble2.best_estimator_
        resultENS2 = (votingEnsemble2.best_score_)
        ensemblePredict = votingEnsemble2.predict(
            normalizedFeatures).reshape(-1, 1)
        print("Voting Forest 2")
        print("Acc and f1: "+str((accuracy, f1)))
        print("Parameters: " + str(votingEnsemble2.best_params_) + '\n')

        parameters = {'criterion': ['entropy'],
                      'max_depth': range(2, 100), }  # 'max_leaf_nodes': range(4, 5)}
        clf = GridSearchCV(DecisionTreeClassifier(random_state=23), parameters, verbose=0,
                           cv=10,  n_jobs=8, refit="f1", scoring=['accuracy', 'f1']).fit(X=ensemblePredict, y=target)
        i = clf.best_index_
        f1 = round(clf.cv_results_['mean_test_f1'][i], 3)
        accuracy = round(clf.cv_results_['mean_test_accuracy'][i], 3)
        entropyDT = clf.best_estimator_
        resultDTE = (clf.best_score_)
        print("Decision Tree Entropy")
        print("Acc and f1: "+str((accuracy, f1)))
        print("Parameters: " + str(clf.best_params_) + '\n')

    def main(self):
        print("hello")
        self.predict()


if __name__ == "__main__":
    predict = Predict("ecoli.csv", "ecoli_test.csv")
    predict.main()
