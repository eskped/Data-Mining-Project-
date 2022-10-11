import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


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

    def standardize(self):
        # print(self.df_numerical.select_dtypes(include=[np.number]).min())
        numerical_data = self.df_numerical.copy()
        scaler = preprocessing.StandardScaler().fit(numerical_data)
        standardized_numerical_data = scaler.transform(numerical_data)
        self.df_numerical_standardized = pd.DataFrame(
            standardized_numerical_data, columns=numerical_data.columns)
        # print(self.df_numerical_standardized.select_dtypes(
        #     include=[np.number]).min())

    def min_max_normalize(self):
        # print(self.df_numerical.select_dtypes(include=[np.number]).min())
        numerical_data = self.df_numerical.copy()
        scaler = preprocessing.MaxAbsScaler().fit(numerical_data)
        normalized_data = scaler.transform(numerical_data)
        self.df_numerical_min_max_normalized = pd.DataFrame(
            normalized_data, columns=numerical_data.columns)
        # print(self.df_min_max_normalized.select_dtypes(
        #     include=[np.number]).min())

    def robust_standardize(self):
        # print(self.df_numerical.select_dtypes(include=[np.number]).max())
        numerical_data = self.df_numerical.copy()
        tup = (0.05, 0.95)
        scaler = preprocessing.RobustScaler(
            quantile_range=tup).fit(numerical_data)
        normalized_data = scaler.transform(numerical_data)
        self.df_numerical_robust_normalized = pd.DataFrame(
            normalized_data, columns=numerical_data.columns)
        # print(self.df_robust_normalized.select_dtypes(
        #     include=[np.number]).max())

    def std3_outlier_processing(self):
        # print(self.df_numerical.select_dtypes(include=[np.number]).max())
        numerical_data = self.df_numerical.copy()
        df_numerical_std3_to_NaN = self.df_numerical.copy()
        df_numerical_std3_to_mean = self.df_numerical.copy()
        df_numerical_std3_to_std = self.df_numerical.copy()
        df_numerical_std3_delete = self.df_numerical.copy()

        for col in numerical_data.columns:
            mean = numerical_data[col].mean()
            std = numerical_data[col].std()
            median = numerical_data[col].median()
            teller = 0
            for x in numerical_data[col]:
                if abs(x - median) > std:
                    teller += 1
                    df_numerical_std3_to_NaN[col] = numerical_data[col].replace(
                        x, np.nan)
                    df_numerical_std3_to_mean[col] = numerical_data[col].replace(
                        x, mean)
                    df_numerical_std3_to_std[col] = numerical_data[col].replace(
                        x, (x - mean) / std)

        self.df_numerical_std3_to_NaN = df_numerical_std3_to_NaN
        self.df_numerical_std3_to_mean = df_numerical_std3_to_mean
        self.df_numerical_std3_to_std = df_numerical_std3_to_std
        # print(self.df_numerical_std3_to_NaN.select_dtypes(
        #     include=[np.number]).max())

    def mean_and_median_imputation(self):
        self.df_numerical_mean_imputed = self.df_numerical.copy().fillna(
            self.df_numerical.mean())
        self.df_numerical_median_imputed = self.df_numerical.copy().fillna(
            self.df_numerical.median())

        nominal_data = self.df_nominal.copy()
        for col in self.df_nominal.copy:
            # mode counts the most frequent value
            ma = nominal_data[col].mode()[0]
            nominal_data[col] = nominal_data[col].fillna(ma)
        self.df_nominal_mean_imputed = nominal_data

    def main(self):

        # Normalization methods
        self.standardize()
        self.min_max_normalize()
        self.robust_standardize()

        # Outlier processing methods
        # Converts the outliers to 'NaN', 'mean' and 'standardization' '
        # delete' not implemented yet
        self.std3_outlier_processing()

        # NaN imputation methods
        self.mean_imputation()
        self.median_imputation()


if __name__ == '__main__':
    solver = Solver('ecoli.csv', 'ecoli_tect.csv')
    solver.main()
