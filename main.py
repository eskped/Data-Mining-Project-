import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from preprocessing import *


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

    def decision_tree_classifier(self):
        # numerical_data = self.df_multivariate_imputed.copy()
        # nominal_data = self.df_nominal_multivariate_imputed.copy()
        numerical_data = self.df_numerical_mean_imputed.copy()
        nominal_data = self.df_nominal_mean_imputed.copy()
        training_data = pd.concat([numerical_data, nominal_data], axis=1)
        training_data = training_data.dropna()
        labels = np.asarray(self.df_target[self.target_column])
        X_train, X_test, y_train, y_test = train_test_split(
            training_data, labels, test_size=0.3, random_state=2)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print(y_pred.shape)
        print(f1_score(y_test, y_pred, average=None))

    def cross_validation(self, model, _X, _y, _cv):
        # numerical_data = self.df_numerical_mean_imputed.copy()
        # nominal_data = self.df_nominal_mean_imputed.copy()
        # _X = pd.concat([numerical_data, nominal_data], axis=1)
        # _y = self.df_target.copy()
        # _cv = 5

        _scoring = ['accuracy', 'precision', 'recall', 'f1']
        results = cross_validate(estimator=model,
                                 X=_X,
                                 y=_y,
                                 cv=_cv,
                                 scoring=_scoring,
                                 return_train_score=True)

        return {"Training Accuracy scores": results['train_accuracy'],
                "Mean Training Accuracy": results['train_accuracy'].mean()*100,
                "Training Precision scores": results['train_precision'],
                "Mean Training Precision": results['train_precision'].mean(),
                "Training Recall scores": results['train_recall'],
                "Mean Training Recall": results['train_recall'].mean(),
                "Training F1 scores": results['train_f1'],
                "Mean Training F1 Score": results['train_f1'].mean(),
                "Validation Accuracy scores": results['test_accuracy'],
                "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
                "Validation Precision scores": results['test_precision'],
                "Mean Validation Precision": results['test_precision'].mean(),
                "Validation Recall scores": results['test_recall'],
                "Mean Validation Recall": results['test_recall'].mean(),
                "Validation F1 scores": results['test_f1'],
                "Mean Validation F1 Score": results['test_f1'].mean()
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

    def model_training(self):
        X = self.df_multivariate_imputed.copy()
        labels = self.df_target.copy()
        decision_tree_model = DecisionTreeClassifier(criterion="entropy", min_samples_split=2, max_depth=1,
                                                     random_state=0)

        decision_tree_result = self.cross_validation(
            decision_tree_model, X, labels, 5)
        print(decision_tree_result)

        model_name = "Decision Tree"
        self.plot_result(model_name,
                         "F1",
                         "F1 Scores in 5 Folds",
                         decision_tree_result["Training F1 scores"],
                         decision_tree_result["Validation F1 scores"], 5)

    def F1_score(self, y_test, y_pred):
        return f1_score(y_test, y_pred, average=None)

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
        self.df_numerical_mean_imputed, self.df_numerical_median_imputed, self.df_nominal_mean_imputed = mean_and_median_imputation(
            numerical_data, nominal_data)
        self.df_numerical_multivariate_imputed, self.df_nominal_multivariate_imputed, self.df_multivariate_imputed = multivariate_imputation(
            self.df.copy(), numerical_data, nominal_data)

        # self.decision_tree_classifier()

        self.model_training()


if __name__ == '__main__':
    solver = Solver('ecoli.csv', 'ecoli_test.csv')
    solver.main()
