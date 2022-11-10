import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate


class Predict:

    def __init__(self, filenameData, filenameTestdata):
        self.df_data = pd.read_csv(filenameData)
        self.df_data_test = pd.read_csv(filenameTestdata)
        self.df_processed, self.df_target = self.preprocessing(self.df_data)
        self.df_normalized_test = self.test_preprocessing(
            self.df_data_test)

    def preprocessing(self, data):
        # Class specific preprocessing
        data1 = data.loc[data['Target (Col 107)'] == 1]
        data0 = data.loc[data['Target (Col 107)'] == 0]

        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        nominalData0 = pd.DataFrame(imputer.fit_transform(
            data0.iloc[:, 103:]), index=data0.iloc[:, 103:].index, columns=data0.iloc[:, 103:].columns)
        numericalData0 = data0.iloc[:, :103].interpolate(
            axis=0, limit_direction='both', method='linear',)
        data0 = pd.concat([numericalData0, nominalData0], axis=1, join="inner")

        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        nominalData1 = pd.DataFrame(imputer.fit_transform(
            data1.iloc[:, 103:]), index=data1.iloc[:, 103:].index, columns=data1.iloc[:, 103:].columns)
        numericalData1 = data1.iloc[:, :103].interpolate(
            axis=0, limit_direction='both', method='linear',)
        data1 = pd.concat([numericalData1, nominalData1], axis=1, join="inner")

        data = pd.concat([data0, data1], axis=0, join="inner")
        normalScaler = MinMaxScaler()
        return pd.DataFrame(
            normalScaler.fit_transform(data.iloc[:, :106]), columns=data.iloc[:, :106].columns), data.iloc[:, 106]

    def test_preprocessing(self, data):
        normalScaler = MinMaxScaler()
        return pd.DataFrame(
            normalScaler.fit_transform(data), columns=data.columns)

    def predict(self):
        df = self.df_processed
        df_labels = self.df_target

        # Gini tree
        giniDT = DecisionTreeClassifier(
            criterion='gini', max_depth=2, max_leaf_nodes=3, random_state=38).fit(df, df_labels)

        # Gini forest
        forest = RandomForestClassifier(
            n_estimators=150, criterion='gini', max_depth=20, random_state=14).fit(df, df_labels)

        # k-NN
        knn = KNeighborsClassifier(
            n_neighbors=9, p=2, weights='uniform').fit(df, df_labels)

        # Voting ensemble
        voting = VotingClassifier(
            estimators=[('giniDT', giniDT), ('forest', forest), ('knn', knn)], voting='hard', n_jobs=8, weights=[1, 2, 2]).fit(df, df_labels)

        ensemblePredict = voting.predict(
            df).reshape(-1, 1)
        ensemblePredictTest = voting.predict(
            self.df_normalized_test).reshape(-1, 1)

        # Entropy tree on prediced values
        entropyDT = DecisionTreeClassifier(
            criterion='entropy', max_depth=2, random_state=23).fit(ensemblePredict, df_labels)
        score = cross_validate(entropyDT,
                               cv=10,  n_jobs=8, scoring=['accuracy', 'f1'], X=ensemblePredict, y=df_labels)
        f1 = round(score['test_f1'].mean(), 3)
        acc = round(score['test_accuracy'].mean(), 3)

        predictedLabels = entropyDT.predict(
            ensemblePredictTest)
        np.savetxt('s4761372.csv', predictedLabels, newline=",\n",
                   fmt="%d", delimiter=",")
        file = open('predictedLabels.csv', 'a')
        file.write(str(acc) + "," + str(f1))
        file.close()

    def main(self):
        self.predict()


if __name__ == "__main__":
    predict = Predict("ecoli.csv", "ecoli_test.csv")
    predict.main()
