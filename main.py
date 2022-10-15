import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from preprocessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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
        random_forest_model = RandomForestClassifier(criterion="gini", max_features=50,
                                                     max_depth=20, n_estimators=500)

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
        results = {}
        for k in range(5, 100):
            k_nearst_neighbor_model = KNeighborsClassifier(n_neighbors=k)
            k_nearst_neighbor_result = self.cross_validation(
                k_nearst_neighbor_model, training_data, labels.values.ravel(), cv)
            results[k] = k_nearst_neighbor_result["Mean Validation F1 Score"]
        print(results)
        if plot:
            model_name = "K Nearst Neighbor"
            self.plot_result(model_name,
                             "F1",
                             "F1 Scores in 5 Folds with K Nearst Neighbor",
                             k_nearst_neighbor_result["Training F1 scores"],
                             k_nearst_neighbor_result["Validation F1 scores"], cv)

        return results

    def main(self):

        # Normalization methods
        numerical_data = self.df_numerical.copy()
        nominal_data = self.df_nominal.copy()
        self.df_numerical_standardized = standardize(numerical_data)
        self.df_numerical_min_max_normalized = min_max_normalize(
            numerical_data)
        self.df_numerical_robust_normalized = robust_standardize(
            numerical_data)

        # Outlier processing methods
        # Converts the outliers to 'NaN', 'mean' and 'standardization'
        self.df_numerical_std3_to_NaN, self.df_numerical_std3_to_mean, self.df_numerical_std3_to_std, self.df_numerical_10_to_NaN, self.df_numerical_10_to_mean, self.df_numerical_10_to_std = std3_and_10_outlier_processing(
            numerical_data)

        # NaN imputation methods
        self.df_numerical_mean_imputed, self.df_numerical_median_imputed, self.df_nominal_median_imputed = mean_and_median_imputation(
            numerical_data, nominal_data)
        self.df_numerical_multivariate_imputed, self.df_nominal_multivariate_imputed, self.df_multivariate_imputed = multivariate_imputation(
            self.df.copy(), numerical_data, nominal_data)

        """ Results from decision tree classifier:
        Rersults of range 0.5 - 0.75 when using numerical and nominal seperately. Max-depth is best around 4
        When using df multivarite results of 1.0
        """
        # print(self.decision_tree_classifier(None,
        #                                     self.df_numerical_mean_imputed, self.df_nominal_median_imputed, 10, True))
        # print(self.random_forest_classifier(None,
        #                                     self.df_numerical_mean_imputed, self.df_nominal_median_imputed, 10, True))

        print(self.k_nearst_neighbor_classifier(None,
                                                self.df_numerical_mean_imputed, self.df_nominal_median_imputed, 10, False))


if __name__ == '__main__':
    solver = Solver('ecoli.csv', 'ecoli_test.csv')
    solver.main()
