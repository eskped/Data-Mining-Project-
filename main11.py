import sys
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from preprocessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics, tree, ensemble, neighbors, linear_model
import pickle
import math


class Solver:

    def __init__(self, data_filename, test_data_filename):
        self.df = pd.read_csv(data_filename)
        self.test_df = pd.read_csv(test_data_filename)
        self.f1_sore = -sys.maxsize
        self.nominal_columns = [
            'Nom (Col 104)', 'Nom (Col 105)', 'Nom (Col 106)']
        self.target_column = ['Target (Col 107)', ]
        self.df_nominal = self.df[self.nominal_columns]
        self.df_numerical = self.df.drop(
            self.nominal_columns, axis=1)
        if len(self.df.columns) == 107:
            self.df_target = self.df[self.target_column]
            self.df_numerical = self.df_numerical.drop(
                self.target_column, axis=1)
        self.df_numericals_processed = []
        self.df_nominals_processed = []

    def cross_validation(self, model, _X, _y, _cv):

        results = cross_validate(estimator=model,
                                 X=_X,
                                 y=_y,
                                 cv=_cv,
                                 scoring=['f1'],
                                 return_train_score=True)

        return {
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean(),
        }

    def plot_result(self, x_label, y_label, plot_title, train_data, val_data, cv):
        plt.figure(figsize=(12, 6))
        labels = []
        for i in range(cv):
            labels.append(f'{i}. Fold')
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

    def decision_tree_classifier(self, training_data, nominal_data, numerical_data, cv, plot):
        if training_data == None:
            training_data = pd.concat([numerical_data, nominal_data], axis=1)
        labels = self.df_target.copy()
        decision_tree_model = DecisionTreeClassifier(criterion="entropy", min_samples_split=10, max_depth=4,
                                                     random_state=None)
        decision_tree_model.fit(training_data, labels)

        decision_tree_result = self.cross_validation(
            decision_tree_model, training_data, labels, cv)
        if plot:
            model_name = "Decision Tree"
            self.plot_result(model_name,
                             "F1",
                             "F1 Scores in 5 Folds",
                             decision_tree_result["Training F1 scores"],
                             decision_tree_result["Validation F1 scores"], cv)

        return decision_tree_result

    def random_forest_classifier(self, training_data, nominal_data, numerical_data, cv, plot):
        if training_data == None:
            training_data = pd.concat([numerical_data, nominal_data], axis=1)
        labels = self.df_target.copy()

        random_forest_model = RandomForestClassifier(
            max_depth=20, max_features=50, n_estimators=51)
        random_forest_model.fit(training_data, labels.values.ravel())
        random_forest_result = self.cross_validation(
            random_forest_model, training_data, labels.values.ravel(), cv)
        if plot:
            model_name = "Random Forest"
            self.plot_result(model_name,
                             "F1",
                             "F1 Scores in 5 Folds with random forest",
                             random_forest_result["Training F1 scores"],
                             random_forest_result["Validation F1 scores"], cv)

        return random_forest_result

    def k_nearst_neighbor_classifier(self, training_data, nominal_data, numerical_data, cv, plot):
        if training_data == None:
            training_data = pd.concat([numerical_data, nominal_data], axis=1)
        labels = self.df_target.copy()
        k_nearst_neighbor_model = KNeighborsClassifier(
            algorithm='auto', n_neighbors=30, p=3, weights='uniform')
        k_nearst_neighbor_model.fit(training_data, labels.values.ravel())
        k_nearst_neighbor_result = self.cross_validation(
            k_nearst_neighbor_model, training_data, labels.values.ravel(), cv)
        if plot:
            model_name = "K Nearst Neighbor"
            self.plot_result(model_name,
                             "F1",
                             "F1 Scores in 5 Folds with K Nearst Neighbor",
                             k_nearst_neighbor_result["Training F1 scores"],
                             k_nearst_neighbor_result["Validation F1 scores"], cv)

        return k_nearst_neighbor_result

    def naive_bayes_classifier(self, nominal_data, numerical_data, cv, plot):
        training_data = pd.concat([numerical_data, nominal_data], axis=1)
        labels = self.df_target.copy()
        naive_bayes_model = GaussianNB(var_smoothing=1)
        naive_bayes_model.fit(training_data, labels.values.ravel())
        naive_bayes_result = self.cross_validation(
            naive_bayes_model, training_data, labels.values.ravel(), cv)
        if plot:
            model_name = "Naive Bayes"
            self.plot_result(model_name,
                             "F1",
                             "F1 Scores in 5 Folds with Naive Bayes",
                             naive_bayes_result["Training F1 scores"],
                             naive_bayes_result["Validation F1 scores"], cv)

        return naive_bayes_result

    def hyper_parameter_tuning(self, data):

        # classifier = DecisionTreeClassifier()
        # parameters = {
        #     # 'criterion': ['gini', 'entropy', 'log_loss'],
        #     'splitter': ['best', 'random'],
        #     'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        #     # 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        #     'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #     # 'max_features': ['sqrt', 'log2', None, 5, 10, 15, 25],
        #     'max_leaf_nodes': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #     # 'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        #     # 'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # }
        # training_data = pd.concat([numerical_data, nominal_data], axis=1)
        parameters = {'criterion': ['entropy'],
                      'max_depth': range(1, 15), 'max_leaf_nodes': range(2, 100)}
        # parameters = {'criterion': ['entropy', 'gini], 'max_depth': range(1, 15)}
        clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters,
                           cv=10,  n_jobs=8, refit="accuracy", scoring=['accuracy', 'f1'])
        clf.fit(X=data, y=self.df_target)
        # clf.predict(self.test_df)
        i = clf.best_index_
        accuracy = math.floor(clf.best_score_ * 1000)/1000.0
        f1 = clf.cv_results_['mean_test_f1'][i]
        combinedTestResults = (accuracy, f1)
        entDT = clf.best_estimator_
        resultDTE = (clf.best_score_)
        print("Decision Tree Entropy")
        print("CTR: "+str(combinedTestResults))
        print(clf.best_score_, clf.best_params_)

    def main(self):

        # Normalization methods
        numerical_data = self.df_numerical.copy()
        nominal_data = self.df_nominal.copy()
        self.df_numerical_standardized = standardize(numerical_data)
        self.df_numerical_min_max_normalized = min_max_normalize(
            numerical_data)
        self.df_numerical_robust_normalized = robust_standardize(
            numerical_data)
        self.df_numerical_maxabs_scaled = maxabs(numerical_data)

        # Outlier processing methods
        # Converts the outliers to 'NaN', 'mean' and 'standardization'
        self.df_numerical_std3_to_NaN, self.df_numerical_std3_to_mean, self.df_numerical_std3_to_std, self.df_numerical_10_to_NaN, self.df_numerical_10_to_mean, self.df_numerical_10_to_std = std3_and_10_outlier_processing(
            numerical_data)

        # NaN imputation methods
        self.df_numerical_mean_imputed, self.df_numerical_median_imputed, self.df_nominal_median_imputed = mean_and_median_imputation(
            numerical_data, nominal_data)
        self.df_numerical_standardized_mean_imputed, self.df_numerical_standardized_median_imputed, _ = mean_and_median_imputation(
            self.df_numerical_standardized, nominal_data)
        self.df_numerical_min_max_mean_imputed, self.df_numerical_min_max_median_imputed, _ = mean_and_median_imputation(
            self.df_numerical_min_max_normalized, nominal_data)
        self.df_numerical_10_to_mean_imputed, self.df_numerical_10_to_median_imputed, _ = mean_and_median_imputation(
            self.df_numerical_10_to_std, nominal_data)
        self.df_numerical_robust_mean_imputed, self.df_numerical_robust_median_imputed, _ = mean_and_median_imputation(
            self.df_numerical_robust_normalized, nominal_data)
        self.df_numerical_maxabs_mean_imputed, self.df_numerical_maxabs_median_imputed, _ = mean_and_median_imputation(
            self.df_numerical_maxabs_scaled, nominal_data)

        numerical_imputed = {
            "df_numerical_mean_imputed": self.df_numerical_mean_imputed,
            "df_numerical_median_imputed": self.df_numerical_median_imputed,
            "df_numerical_standardized_mean_imputed": self.df_numerical_standardized_mean_imputed,
            "df_numerical_standardized_median_imputed": self.df_numerical_standardized_median_imputed,
            "df_numerical_min_max_mean_imputed": self.df_numerical_min_max_mean_imputed,
            "df_numerical_min_max_median_imputed": self.df_numerical_min_max_median_imputed,
            "df_numerical_10_to_mean_imputed": self.df_numerical_10_to_mean_imputed,
            "df_numerical_10_to_median_imputed": self.df_numerical_10_to_median_imputed,
            "df_numerical_robust_mean_imputed": self.df_numerical_robust_mean_imputed,
            "df_numerical_robust_median_imputed": self.df_numerical_robust_median_imputed,
            "df_numerical_maxabs_mean_imputed": self.df_numerical_maxabs_mean_imputed,
            "df_numerical_maxabs_median_imputed": self.df_numerical_maxabs_median_imputed,
        }

        df_class_imputed, df_numerical_class_imputed, df_nominal_class_imputed = normalize(
            self.df.copy())
        # print(df_numerical_class_imputed)
        # print(df_nominal_class_imputed)

        # for key, value in numerical_imputed.items():
        #     value.to_csv(key + ".csv")

        # with open('numerical_imputed.pkl', 'wb') as f:
        #     pickle.dump(numerical_imputed, f)

        # # with open('numerical_imputed.pkl', 'rb') as f:
        # #     loaded_dict = pickle.load(f)

        # self.df_nominal_median_imputed.to_csv(
        #     "df_nominal_median_imputed.csv")

        """ Results from decision tree classifier:
        df_numerical_median_imputed: CV score=0.793. {'criterion': 'entropy', 'max_depth': 5, 'max_leaf_nodes': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'random'}
        df_numerical_mean_imputed: CV score=0.788. Best parameters: {'ccp_alpha': 0.0, 'criterion': 'entropy', 'max_depth': 6, 'max_leaf_nodes': 7, 'splitter': 'random'}
        df_numerical_10_to_mean_imputed: CV score=0.788. Best parameters : {'ccp_alpha': 0.0, 'criterion': 'log_loss', 'max_depth': 9, 'max_leaf_nodes': 6, 'splitter': 'random'}
        df_numerical_10_to_median_imputed: CV score=0.787. Best parameters: {'ccp_alpha': 0.0, 'criterion': 'entropy', 'max_depth': 7, 'max_leaf_nodes': 6, 'splitter': 'random'}
        df_numerical_min_max_median_imputed: CV score=0.785. Best parameters: {'ccp_alpha': 0.0, 'criterion': 'log_loss', 'max_depth': 6, 'max_leaf_nodes': 6, 'splitter': 'random'}
        df_numerical_10_to_mean_imputed: CV score=0.773. Best parameters: {'criterion': 'log_loss', 'max_depth': 3, 'max_features': None, 'min_impurity_decrease': 0.0, 'splitter': 'random'}
        df_numerical_robust_mean_imputed: CV score=0.791. Best parameters from gridsearch: {'criterion': 'entropy', 'max_depth': 10, 'max_leaf_nodes': 6, 'min_samples_leaf': 1, 'min_samples_split': 4, 'splitter': 'random'}
        

        """
        # print(self.decision_tree_classifier(None,
        #                                     self.df_numerical_min_max_median_imputed, self.df_nominal_median_imputed, 10, True))
        """Results from random forest classifier:
        df_numerical_min_max_mean_imputed: CV score=0.784. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 10, 'max_features': 25, 'n_estimators': 201}
        df_numerical_mean_imputed: CV score=0.778. Best parameters from gridsearch: {'criterion': 'entropy', 'max_depth': 5, 'max_features': 50, 'n_estimators': 201}. 
        df_numerical_median_imputed: CV score=0.782. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 5, 'max_features': 50, 'n_estimators': 51}
        df_numerical_standardized_mean_imputed: score=0.781. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 10, 'max_features': 50, 'n_estimators': 201}
        df_numerical_standardized_median_imputed: CV score=0.780. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 10, 'max_features': 50, 'n_estimators': 201}
        df_numerical_min_max_median_imputed: CV score=0.780. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 10, 'max_features': 50, 'n_estimators': 201}
        df_numerical_10_to_mean_imputed: CV score=0.781. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 3, 'max_features': 50, 'n_estimators': 101}
        df_numerical_10_to_median_imputed: CV score=0.779. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 6, 'max_features': 50, 'n_estimators': 501}
        df_numerical_robust_mean_imputed: CV score=0.783. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 10, 'max_features': 25, 'n_estimators': 501}
        df_numerical_robust_median_imputed: CV score=0.781. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 10, 'max_features': 50, 'n_estimators': 51}
        df_numerical_maxabs_mean_imputed: CV score=0.782. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 8, 'max_features': 50, 'n_estimators': 201}
        df_numerical_maxabs_median_imputed: CV score=0.779. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 9, 'max_features': 50, 'n_estimators': 101}
        """

        # print(self.random_forest_classifier(None,
        #                                     self.df_numerical_mean_imputed, self.df_nominal_median_imputed, 10, False))

        """Results from K Nearst Neighbor classifier:
        Use df_numerical_min_max_[]_imputed, df_numerical_maxabs_[]_imputed
        Results from 0.6-0.85. Mean 0.768
        Hyperparameters: algorithm': 'auto', 'n_neighbors': 30, 'p': 3, 'weights': 'uniform'
        """
        # print(self.k_nearst_neighbor_classifier(None,
        #                                         self.df_numerical_min_max_median_imputed, self.df_nominal_median_imputed, 10, False))

        """ results from Naive Bayes classifier:
        numerical data: df_numerical_min_max_median_imputed, df_numerical_maxabs_mean_imputed or df_numerical_maxabs_median_imputed
        results: 0.766
        Hyper: var_smooting: 1
        """
        # print(self.naive_bayes_classifier(
        #     self.df_numerical_maxabs_median_imputed, self.df_nominal_median_imputed, 10, False))

        # for _, key in enumerate(numerical_imputed):
        #     print(f'Numerical data is: {key}')
        self.hyper_parameter_tuning(
            df_class_imputed)

        # print(numerical_imputed[key])

        # self.hyper_parameter_tuning(
        #     self.df_numerical_maxabs_median_imputed, self.df_nominal_median_imputed)


if __name__ == '__main__':
    solver = Solver('ecoli.csv', 'ecoli_test.csv')
    solver.main()
