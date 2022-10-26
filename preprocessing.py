from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, maxabs_scale


def standardize(numerical_data):
    scaler = preprocessing.StandardScaler().fit(numerical_data)
    standardized_numerical_data = scaler.transform(numerical_data)
    return pd.DataFrame(
        standardized_numerical_data, columns=numerical_data.columns)


def maxabs(numerical_data):
    return pd.DataFrame(maxabs_scale(numerical_data), columns=numerical_data.columns)


def min_max_normalize(numerical_data):
    scaler = preprocessing.MaxAbsScaler().fit(numerical_data)
    normalized_data = scaler.transform(numerical_data)
    return pd.DataFrame(
        normalized_data, columns=numerical_data.columns)


def robust_standardize(numerical_data):
    tup = (0.05, 0.95)
    scaler = preprocessing.RobustScaler(
        quantile_range=tup).fit(numerical_data)
    normalized_data = scaler.transform(numerical_data)
    return pd.DataFrame(
        normalized_data, columns=numerical_data.columns)


def std3_and_10_outlier_processing(numerical_data):
    df_numerical_std3_to_NaN = numerical_data.copy()
    df_numerical_std3_to_mean = numerical_data.copy()
    df_numerical_std3_to_std = numerical_data.copy()

    df_numerical_10_to_NaN = numerical_data.copy()
    df_numerical_10_to_mean = numerical_data.copy()
    df_numerical_10_to_std = numerical_data.copy()

    for col in numerical_data.columns:
        mean = numerical_data[col].mean()
        std = numerical_data[col].std()
        median = numerical_data[col].median()
        for x in numerical_data[col]:
            if abs(x - median) > std:
                df_numerical_std3_to_NaN[col] = numerical_data[col].replace(
                    x, np.nan)
                df_numerical_std3_to_mean[col] = numerical_data[col].replace(
                    x, mean)
                df_numerical_std3_to_std[col] = numerical_data[col].replace(
                    x, (x - mean) / std)
            if abs(x) > 5:
                df_numerical_10_to_NaN[col] = numerical_data[col].replace(
                    x, np.nan)
                df_numerical_10_to_mean[col] = numerical_data[col].replace(
                    x, mean)
                df_numerical_10_to_std[col] = numerical_data[col].replace(
                    x, (x - mean) / std)

    return df_numerical_std3_to_NaN, df_numerical_std3_to_mean, df_numerical_std3_to_std, df_numerical_10_to_NaN, df_numerical_10_to_mean, df_numerical_10_to_std


def mean_and_median_imputation(numerical_data, nominal_data):
    for col in nominal_data:
        # mode counts the most frequent value
        ma = nominal_data[col].mode()[0]
        nominal_data[col] = nominal_data[col].fillna(ma)

    return numerical_data.copy().fillna(
        numerical_data.mean()), numerical_data.copy().fillna(
        numerical_data.median()), nominal_data


def multivariate_imputation(data, numerical_data, nominal_data):
    try:
        return pd.read_csv(
            'df_numerical_multivariate_imputed.csv'), pd.read_csv(
            'df_nominal_multivariate_imputed.csv'), pd.read_csv(
            'df_multivariate_imputed.csv')
    except:
        # numerical
        imputer = IterativeImputer(max_iter=10, random_state=None)
        imputed = imputer.fit_transform(numerical_data)
        df_imputed = pd.DataFrame(imputed, columns=numerical_data.columns)
        df_imputed.to_csv('df_numerical_multivariate_imputed.csv')
        # nominal
        imp = IterativeImputer(max_iter=10, random_state=None)
        imp = imp.fit_transform(nominal_data)
        df_imp = pd.DataFrame(imp, columns=nominal_data.columns)
        df_imp.to_csv('df_nominal_multivariate_imputed.csv')
        # all
        imput = IterativeImputer(max_iter=10, random_state=None)
        imput = imput.fit_transform(data)
        df_imput = pd.DataFrame(imput, columns=data.columns)
        df_imput.to_csv('df_multivariate_imputed.csv')

        df_imputed = pd.read_csv('df_numerical_multivariate_imputed.csv')
        df_imp = pd.read_csv('df_nominal_multivariate_imputed.csv')

        return df_imputed, df_imp, df_imput


def normalize(data):

    data0 = data.loc[data['Target (Col 107)'] == 0]
    data1 = data.loc[data['Target (Col 107)'] == 1]

    numericalData0 = data0.iloc[:, :103]
    nominalData0 = data0.iloc[:, 103:]

    numericalData0 = numericalData0.interpolate(
        method='linear', axis=0, limit_direction='both')
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(nominalData0)
    nominalData0 = pd.DataFrame(imputer.transform(
        nominalData0), index=nominalData0.index)

    nominalData0.rename({0: 'Nom (Col 104)',
                         1: 'Nom (Col 105)',
                         2: 'Nom (Col 106)',
                         3: 'Target (Col 107)'}, axis=1, inplace=True)

    data0 = pd.concat([numericalData0, nominalData0], axis=1, join="inner")

    numericalData1 = data1.iloc[:, :103]
    nominalData1 = data1.iloc[:, 103:]

    numericalData1 = numericalData1.interpolate(
        method='linear', axis=0, limit_direction='both')
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(nominalData1)
    nominalData1 = pd.DataFrame(imputer.transform(
        nominalData1), index=nominalData1.index)

    nominalData1.rename({0: 'Nom (Col 104)',
                         1: 'Nom (Col 105)',
                         2: 'Nom (Col 106)',
                         3: 'Target (Col 107)'}, axis=1, inplace=True)

    data1 = pd.concat([numericalData1, nominalData1], axis=1, join="inner")

    data = pd.concat([data0, data1], axis=0, join="inner")

    normalScaler = MinMaxScaler()
    normalizedData = pd.DataFrame(
        normalScaler.fit_transform(data), columns=data.columns)

    return normalizedData, normalizedData.iloc[:, :103], normalizedData.iloc[:, 103:106]
