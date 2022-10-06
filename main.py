import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


class Solver:

    def __init__(self, data_filename, test_data_filename):
        self.df = pd.read_csv(data_filename)
        self.test_df = pd.read_csv(test_data_filename)
        self.f1_sore = -sys.maxsize

    def normalize(self):
        data = self.df.copy()
        scaler = preprocessing.StandardScaler().fit(data)
        normalized_data = scaler.transform(data)
        self.normalized_df = pd.DataFrame(
            normalized_data, columns=data.columns)
        print(self.normalized_df['Num (Col 2)'][529])

    def min_max_standardize(self):
        data = self.df.copy()
        scaler = preprocessing.MaxAbsScaler().fit(data)
        normalized_data = scaler.transform(data)
        self.min_max_normalized_df = pd.DataFrame(
            normalized_data, columns=data.columns)
        print(self.min_max_normalized_df['Num (Col 2)'][529])

    def robust_standardize(self):
        data = self.df.copy()
        scaler = preprocessing.RobustScaler().fit(data)
        normalized_data = scaler.transform(data)
        self.robust_normalized_df = pd.DataFrame(
            normalized_data, columns=data.columns)
        print(self.robust_normalized_df['Num (Col 2)'][529])

    def std3_outlier_to_(self, outlier_proceccing_method):
        data = self.df.copy()
        for col in data.columns:
            mean = data[col].mean()
            std = data[col].std()
            for x in data[col]:
                if x > mean + 3 * std or x < mean - 3 * std:
                    if outlier_proceccing_method == 'NaN':
                        data[col] = data[col].replace(x, np.nan)
                    elif outlier_proceccing_method == 'mean':
                        data[col] = data[col].replace(x, mean)
                    elif outlier_proceccing_method == 'standardization':
                        data[col] = data[col].replace(x, (x - mean) / std)
                    elif outlier_proceccing_method == 'delete':
                        data = data.drop(data[data[col] == x].index)
        self.outlier_processed_df = data
        print(self.outlier_processed_df['Num (Col 2)'][529])
        # det er her du er Eskil

    def main(self):

        self.normalize()
        self.min_max_standardize()
        self.robust_standardize()
        # options are 'Nan', 'mean', 'standardization', 'delete'
        self.std3_outlier_to_('standardization')


if __name__ == '__main__':
    solver = Solver('ecoli.csv', 'ecoli_tect.csv')
    solver.main()
