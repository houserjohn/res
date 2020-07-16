#to properly process data before running examples, make sure to run newPreP.py first and then run this file.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#df = pd.read_csv(f'./data/ml-100k/ml-100k/u.data', sep='\t', names=['old_user', 'old_item', 'rating', 'timestamp'], header =None)

df = pd.read_csv(f'./sProcessedData.csv', names=[ 'item', 'rating', 'timestamp','user'], header =None)

def remove2letters(item):
    return item[2:]

df['user'] = df.user.apply(remove2letters)
df['item'] = df.item.apply(remove2letters)


#l = list(df.old_user.unique())
l = list(df.user.unique())
d = dict(zip(l, range(len(l))))
#df['user'] = df.old_user.map(d)
df['user'] = df.user.map(d)


#l = list(df.old_item.unique())
l = list(df.item.unique())
d = dict(zip(l, range(len(l))))
#df['item'] = df.old_item.map(d)
df['item'] = df.item.map(d)

df_train, df_test = train_test_split(df, test_size=0.10, random_state=42)

df_train.to_csv('u.traindata', index=False)

df_test.to_csv('u.testdata', index=False)

