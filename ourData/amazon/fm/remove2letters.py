#removes first letters of item and user columns

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#df = pd.read_csv(f'./data/ml-100k/ml-100k/u.data', sep='\t', names=['old_user', 'old_item', 'rating', 'timestamp'], header =None)

df = pd.read_csv(f'./sTestData.csv', names=['blank', 'item', 'rating', 'text','user'], header =None)

def remove2letters(item):
    return item[2:]

df['user'] = df.user.apply(remove2letters)
df['item'] = df.item.apply(remove2letters)

df.to_csv('u.testdata', index=False)


