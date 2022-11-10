import sys
import numpy as np
import csv
import math
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, tree, ensemble, neighbors, linear_model
import pickle
import sys
import os
import time

# Disable


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore


blockPrint()


def main():
    contents = pd.read_csv('Ecoli.csv', header=0)
    testContents = pd.read_csv('Ecoli_test.csv', header=0)

    cdf = contents
    testContents = testContents
    # print(cdf)
    # print(testContents)
    # 1500x1
    # print(cdf)
    cdf0 = cdf.loc[cdf['Target (Col 107)'] == 0]
    # print(cdf0)

    print("cdf0 contains NaN " + str(cdf0.isnull().values.any()))
    cdf1 = cdf.loc[cdf['Target (Col 107)'] == 1]
    # print(cdf1)
    print("cdf1 contains NaN " + str(cdf1.isnull().values.any()))

    continuousCdf0 = cdf0.iloc[:, :103]
    # print(continuousCdf)
    binaryCdf0 = cdf0.iloc[:, 103:]

    continuousCdf0 = continuousCdf0.interpolate(
        method='linear', axis=0, limit_direction='both')
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(binaryCdf0)
    binaryCdf0 = pd.DataFrame(imputer.transform(
        binaryCdf0), index=binaryCdf0.index)
    print('continuousCdf0 shape: '+str(continuousCdf0.shape))
    print('binaryCdf0 shape: '+str(binaryCdf0.shape))
    cdf0 = pd.concat([continuousCdf0, binaryCdf0], axis=1, join="inner")

    continuousCdf1 = cdf1.iloc[:, :103]
    # print(continuousCdf)
    binaryCdf1 = cdf1.iloc[:, 103:]

    continuousCdf1 = continuousCdf1.interpolate(
        method='linear', axis=0, limit_direction='both')
    imputer.fit(binaryCdf1)
    binaryCdf1 = pd.DataFrame(imputer.transform(
        binaryCdf1), index=binaryCdf1.index)
    print('continuousCdf1 shape: '+str(continuousCdf1.shape))
    print('binaryCdf1 shape: '+str(binaryCdf1.shape))
    cdf1 = pd.concat([continuousCdf1, binaryCdf1], axis=1, join="inner")

    cdf = pd.concat([cdf0, cdf1], axis=0, join="inner")
    # print(cdf)

    normalScaler = MinMaxScaler()
    normalizedCdf = pd.DataFrame(
        normalScaler.fit_transform(cdf), columns=cdf.columns)
    # testNormal
    # normalScaler = MinMaxScaler()
    # testContents = pd.DataFrame(normalScaler.fit_transform(testContents), columns=testContents.columns)

    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(normalizedCdf)
    # sys.exit()

    normalizedFeatures = normalizedCdf.iloc[:, :106]  # X
    print('normalizedFeatures shape: '+str(normalizedFeatures.shape))
    # print(normalizedFeatures)
    print("normalizedFeatures Contains NaN:")
    print(normalizedFeatures.isnull().values.any())
    normalizedTarget = normalizedCdf.iloc[:, 106]  # y
    print('normalizedTarget shape: '+str(normalizedTarget.shape))
    # print(normalizedTarget)
    print("normalizedTarget Contains NaN:")
    print(normalizedTarget.isnull().values.any())

    # standardizedFeatures = standardizedCdf.iloc[:,:116] #X
    # print('standardizedFeatures shape: '+str(standardizedFeatures.shape))
    # print(standardizedFeatures)
    # print("standardizedFeatures Contains NaN:")
    # print(standardizedFeatures.isnull().values.any())
    # standardizedTarget = standardizedCdf[116] #y
    # print('standardizedTarget shape: '+str(standardizedTarget.shape))
    # print(standardizedTarget)
    # print("standardizedTarget Contains NaN:")
    # print(standardizedTarget.isnull().values.any())
    # sys.exit()

    file = open("output.txt", "w")

    with open('../data/numerical_imputed.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    # print('dict', loaded_dict)

    # for key in loaded_dict:
    # file.write(str(key))
    # file.write('\n')

    # noe = normalizedFeatures.iloc[:, 103:]
    # normalizedFeatures = pd.concat(
    #     [loaded_dict[key], noe], axis=1, join="inner")

    # normalizedFeatures = pd.read_csv('data.csv')

    print('============================')
    print('=========TRAIN DATA=========')
    print('============================')
    # Decision Tree

    parameters = {'criterion': ['entropy'],
                  'max_depth': range(1, 15), 'max_leaf_nodes': range(2, 100)}
    # parameters = {'criterion': ['entropy', 'gini], 'max_depth': range(1, 15)}
    clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters,
                       cv=10,  n_jobs=4, refit="accuracy", scoring=['accuracy', 'f1'])
    clf.fit(X=normalizedFeatures, y=normalizedTarget)
    clf.predict(testContents)
    i = clf.best_index_
    accuracy = math.floor(clf.best_score_ * 1000)/1000.0
    f1 = math.floor(clf.cv_results_['mean_test_f1'][i] * 1000)/1000.0
    combinedTestResults = (accuracy, f1)
    entDT = clf.best_estimator_
    resultDTE = (clf.best_score_)
    print("Decision Tree Entropy")
    print("CTR: "+str(combinedTestResults))
    print(clf.best_score_, clf.best_params_)

    # file = open("output.txt", "w")
    file.write("Decision Tree Entropy\n")
    file.write("CTR: "+str(combinedTestResults) + "\n")
    file.write(str(clf.best_score_) + " " + str(clf.best_params_) + "\n\n")
    # file.close()

    parameters = {'max_depth': range(
        1, 15), 'max_leaf_nodes': range(2, 100)}
    clf = GridSearchCV(tree.DecisionTreeClassifier(criterion="gini"), parameters,
                       cv=10,  n_jobs=4, refit="f1", scoring=['accuracy', 'f1'])
    clf.fit(X=normalizedFeatures, y=normalizedTarget)
    testResults = clf.predict(testContents)
    testResults = testResults.tolist()
    i = clf.best_index_
    accuracy = math.floor(clf.best_score_ * 1000)/1000.0
    f1 = math.floor(clf.cv_results_['mean_test_f1'][i] * 1000)/1000.0
    combinedTestResults = (accuracy, f1)
    clf.predict(testContents)
    giniDT = clf.best_estimator_
    print("Decision Tree Gini")
    print("CTR: "+str(combinedTestResults))
    print(str(type(testResults)))

    # file = open("output.txt", "w")
    file.write("Decision Tree Gini\n")
    file.write("CTR: "+str(combinedTestResults) + "\n\n")
    # file.close()

    with open('s4761372.csv', mode='w', newline='') as resultFile:
        fullWriter = csv.writer(resultFile, delimiter=',', quotechar='"',
                                lineterminator=',\r\n', quoting=csv.QUOTE_MINIMAL)
        fullWriter.writerows(map(lambda x: [int(x)], testResults))
    with open('s4761372.csv', mode='a', newline='') as resultFile:
        fullWriter = csv.writer(resultFile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fullWriter.writerow(combinedTestResults)

    # random forest

    parameters = {'max_depth': range(1, 10)}
    clf = GridSearchCV(ensemble.RandomForestClassifier(class_weight="balanced", random_state=6,
                                                       criterion="entropy"), parameters, n_jobs=4, refit="f1", scoring=['accuracy', 'f1'])
    clf.fit(X=normalizedFeatures, y=normalizedTarget)
    # clf.predict(testContents)
    i = clf.best_index_
    accuracy = math.floor(clf.best_score_ * 1000)/1000.0
    f1 = math.floor(clf.cv_results_['mean_test_f1'][i] * 1000)/1000.0
    combinedTestResults = (accuracy, f1)
    entRF = clf.best_estimator_
    print("Random Forest Entropy")
    print(clf.best_score_, clf.best_params_)
    print("CTR: "+str(combinedTestResults))
    print(str(type(testResults)))

    # file = open("output.txt", "w")
    file.write("Random Forest Entropy\n")
    file.write("CTR: "+str(combinedTestResults) + "\n")
    file.write(str(clf.best_score_) + " " + str(clf.best_params_) + "\n\n")
    # file.close()

    parameters = {'max_depth': range(1, 15)}
    clf = GridSearchCV(ensemble.RandomForestClassifier(class_weight="balanced", random_state=6,
                                                       criterion="gini"), parameters, n_jobs=4, refit="accuracy", scoring=['accuracy', 'f1'])
    clf.fit(X=normalizedFeatures, y=normalizedTarget)
    # clf.predict(testContents)
    i = clf.best_index_
    accuracy = math.floor(clf.best_score_ * 1000)/1000.0
    f1 = math.floor(clf.cv_results_['mean_test_f1'][i] * 1000)/1000.0
    combinedTestResults = (accuracy, f1)
    giniRF = clf.best_estimator_
    print("Random Forest Gini")
    print(clf.best_score_, clf.best_params_)
    print("CTR: "+str(combinedTestResults))
    print(str(type(testResults)))

    # file = open("output.txt", "w")
    file.write("Random Forest Gini\n")
    file.write("CTR: "+str(combinedTestResults) + "\n")
    file.write(str(clf.best_score_) + " " + str(clf.best_params_) + "\n\n")
    # file.close()

    # k-NN

    parameters = {'n_neighbors': np.arange(1, 40), 'p': [1, 2, 3]}
    knn_gscv = GridSearchCV(neighbors.KNeighborsClassifier(
    ), parameters, cv=10, n_jobs=4, refit="f1", scoring=['accuracy', 'f1'])
    knn_gscv.fit(X=normalizedFeatures, y=normalizedTarget)
    # clf.predict(testContents)
    i = knn_gscv.best_index_
    accuracy = math.floor(knn_gscv.best_score_ * 1000)/1000.0
    f1 = math.floor(knn_gscv.cv_results_['mean_test_f1'][i] * 1000)/1000.0
    combinedTestResults = (accuracy, f1)
    knnBest = knn_gscv.best_estimator_
    print("kNN")
    print(knn_gscv.best_score_, knn_gscv.best_params_)
    print("CTR: "+str(combinedTestResults))
    print(str(type(testResults)))

    # file = open("output.txt", "w")
    file.write("kNN\n")
    file.write("CTR: "+str(combinedTestResults) + "\n")
    file.write(str(knn_gscv.best_score_) + " " +
               str(knn_gscv.best_params_) + "\n\n")
    # file.close()

    # naïve bayes

    parameters = {}
    GaussianNBModel = GridSearchCV(GaussianNB(
    ), parameters, cv=10, n_jobs=4, refit="f1", scoring=['accuracy', 'f1'])
    GaussianNBModel.fit(X=normalizedFeatures, y=normalizedTarget)
    # clf.predict(testContents)
    i = GaussianNBModel.best_index_
    accuracy = math.floor(GaussianNBModel.best_score_ * 1000)/1000.0
    f1 = math.floor(GaussianNBModel.cv_results_[
        'mean_test_f1'][i] * 1000)/1000.0
    combinedTestResults = (accuracy, f1)
    GNBBest = GaussianNBModel.best_estimator_
    print('\nGaussian NB')
    print(GaussianNBModel.best_score_)
    print("CTR: "+str(combinedTestResults))
    print(str(type(testResults)))

    # file = open("output.txt", "w")
    file.write("Gaussian NB\n")
    file.write("CTR: "+str(combinedTestResults) + "\n")
    file.write(str(GaussianNBModel.best_score_) + "\n\n")
    # file.close()

    # Ensemble 4 Normalization

    estimators = [('DecisionTree Entropy', entDT), ('DecisionTree Gini', giniDT), ('knn', knnBest),
                  ('Gaussian NB', GNBBest), ('Random Forest Entropy', entRF), ('Random Forest Gini', giniRF)]
    parameters = {'voting': ['hard', 'soft'], }
    votingEnsemble = GridSearchCV(ensemble.VotingClassifier(
        estimators), parameters, cv=10, n_jobs=4, refit="f1", scoring=['accuracy', 'f1'])
    votingEnsemble.fit(normalizedFeatures, normalizedTarget)
    i = votingEnsemble.best_index_
    accuracy = math.floor(votingEnsemble.best_score_ * 1000)/1000.0
    f1 = math.floor(votingEnsemble.cv_results_[
        'mean_test_f1'][i] * 1000)/1000.0
    combinedTestResults = (accuracy, f1)
    # clf.predict(testContents)
    print('\nVoting Ensemble')
    print(votingEnsemble.best_score_)
    print("CTR: "+str(combinedTestResults))
    print(str(type(testResults)))

    # file = open("output.txt", "w")
    file.write("Voting Ensemble\n")
    file.write("CTR: "+str(combinedTestResults) + "\n")
    file.write(str(votingEnsemble.best_score_) + "\n")
    file.write(str(votingEnsemble.best_params_))
    file.write("\n\n")
    file.close()


if __name__ == "__main__":
    main()
