import pandas as pd

main1 = pd.read_csv('targetmain1.csv')
f123 = pd.read_csv('target123.csv')


# print(list(test.isna().sum()))
print(main1.compare(f123))
