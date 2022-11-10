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


data = pd.read_csv('Ecoli.csv')
testContents = pd.read_csv('Ecoli_test.csv')


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

normalizedFeatures.to_csv('normalizedFeatures123.csv')
target.to_csv('target123.csv')

parameters = {'criterion': ['entropy'],
              'max_depth': range(2, 5), 'max_leaf_nodes': range(4, 5)}
clf = GridSearchCV(DecisionTreeClassifier(random_state=14), parameters, verbose=0,
                   cv=10,  n_jobs=8, refit="f1", scoring=['accuracy', 'f1']).fit(X=normalizedFeatures, y=target)
# clf.predict(testContents)
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
# clf.predict(testContents)
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
# clf.predict(testContents)
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
# neigh.predict(testContents)
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


# estimators = [('DecisionTree Entropy', entropyDT), ('DecisionTree Gini', giniDT), ('Random Forest Entropy', forest), ('knn', knn),
#               ('Gaussian NB', gnbBest)]
# parameters = {'voting': ['hard'], }
# votingEnsemble = GridSearchCV(ensemble.VotingClassifier(
#     estimators,), parameters, verbose=0, cv=10, n_jobs=8, refit="f1", scoring=['accuracy', 'f1']).fit(X=normalizedFeatures, y=target)
# i = votingEnsemble.best_index_
# f1 = round(votingEnsemble.cv_results_['mean_test_f1'][i], 3)
# accuracy = round(votingEnsemble.cv_results_['mean_test_accuracy'][i], 3)
# ensemble = votingEnsemble.best_estimator_
# resultENS = (votingEnsemble.best_score_)
# ensemblePredict = votingEnsemble.predict(testContents)
# print("Voting Forest")
# print("Acc and f1: "+str((accuracy, f1)))
# print("Parameters: " + str(votingEnsemble.best_params_) + '\n')


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
ensemblePredict = votingEnsemble2.predict(normalizedFeatures).reshape(-1, 1)
print("Voting Forest 2")
print("Acc and f1: "+str((accuracy, f1)))
print("Parameters: " + str(votingEnsemble2.best_params_) + '\n')


parameters = {'criterion': ['entropy'],
              'max_depth': range(2, 100), }  # 'max_leaf_nodes': range(4, 5)}
clf = GridSearchCV(DecisionTreeClassifier(random_state=23), parameters, verbose=0,
                   cv=10,  n_jobs=8, refit="f1", scoring=['accuracy', 'f1']).fit(X=ensemblePredict, y=target)
# clf.predict(testContents)
i = clf.best_index_
f1 = round(clf.cv_results_['mean_test_f1'][i], 3)
accuracy = round(clf.cv_results_['mean_test_accuracy'][i], 3)
entropyDT = clf.best_estimator_
resultDTE = (clf.best_score_)
print("Decision Tree Entropy")
print("Acc and f1: "+str((accuracy, f1)))
print("Parameters: " + str(clf.best_params_) + '\n')


if False:

    file = open("output predicted.txt", "w")

    bestCombo = [
        [('DecisionTree Entropy', entropyDT), ('knn', knn)],
        [('DecisionTree Gini', giniDT), ('knn', knn)],
        [('DecisionTree Entropy', entropyDT),
         ('Random Forest Entropy', forest), ('knn', knn)],
        [('DecisionTree Gini', giniDT),
         ('Random Forest Entropy', forest), ('knn', knn)],
        [('Random Forest Entropy', forest),
         ('knn', knn), ('Gaussian NB', gnbBest)],
        [('DecisionTree Entropy', entropyDT), ('DecisionTree Gini', giniDT),
         ('Random Forest Entropy', forest), ('knn', forest)],
        [('DecisionTree Gini', giniDT), ('Random Forest Entropy', forest),
         ('knn', knn), ('Gaussian NB', gnbBest)],
    ]
    bestCombo = [list(elem) for elem in bestCombo]

    weights = [1, 1, 1, 1, 1, 1, 1,
               0.25, 0.25, 0.25, 0.25, 0.25,
               0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
               0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
               2, 2, 2, 2, 2, 2, 2]

    for estimator in bestCombo:
        print(estimator)
        weight_combos = []
        file.write(str(estimator) + '\n')
        for subset in itertools.combinations(weights, len(estimator)):
            weight_combos.append(subset)
        weight_combos = list(set(weight_combos))
        weight_combos = [list(elem) for elem in weight_combos]

        bestF1 = 0
        bestAcc = 0
        bestWeights = []
        for weight_combo in weight_combos:
            parameters = {'weights': [weight_combo], 'voting': ['hard'], }

            votingEnsemble2 = GridSearchCV(VotingClassifier(
                estimator), parameters,  verbose=1, cv=10, n_jobs=8, refit="f1", scoring=['accuracy', 'f1'])

            votingEnsemble2.fit(X=ensemblePredict, y=target)
            i = votingEnsemble2.best_index_
            f1 = round(votingEnsemble2.cv_results_['mean_test_f1'][i], 3)
            accuracy = round(votingEnsemble2.cv_results_[
                'mean_test_accuracy'][i], 3)
            ensemble2 = votingEnsemble2.best_estimator_
            resultENS2 = (votingEnsemble2.best_score_)
            ensemblePredict = votingEnsemble2.predict(testContents)
            print("Voting Forest 2")
            print("Acc and f1: "+str((accuracy, f1)))
            print("Parameters: " + str(votingEnsemble2.best_params_) + '\n')

            if f1 > bestF1:
                bestF1 = f1
                bestAcc = accuracy
                bestWeights = weight_combo

        file.write("Best F1: " + str(bestF1) + '\n')
        file.write("Best Acc: " + str(bestAcc) + '\n')
        file.write("Best Weights: " + str(bestWeights) + '\n')
        file.write('\n')

    file.close()
