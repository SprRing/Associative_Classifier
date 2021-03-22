import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from sklearn import preprocessing
				
col_names = ['age', 'income', 'student', 'credit_rating', 'buys_computer']
pima = pd.read_csv("/Users/yenshou/Desktop/Data mining/ACs.csv", header=0, names=col_names)
df= pima.copy()


le = preprocessing.LabelEncoder()
df['credit_rating'] = le.fit_transform(df['credit_rating'])
df['student'] = le.fit_transform(df['student'])
df['buys_computer'] = le.fit_transform(df['buys_computer'])

df_dum = pd.get_dummies(df['income'])
df=pd.concat([df_dum, df[['age', 'student', 'credit_rating', 'buys_computer']]],axis=1)

df_dum = pd.get_dummies(df['age'])
df=pd.concat([df_dum, df[['high', 'medium', 'low', 'student', 'credit_rating', 'buys_computer']]],axis=1)

print(df)
# support = 10% confidence = 60%
itemsets = apriori(df.loc[0:13], min_support=0.1, use_colnames=True)
itemsets['length'] = itemsets['itemsets'].apply(lambda x:len(x))
# print(itemsets)
itemsets = association_rules(itemsets, metric="confidence", min_threshold=0.6)
itemsets.to_csv("/Users/yenshou/Desktop/Data mining/result.csv")