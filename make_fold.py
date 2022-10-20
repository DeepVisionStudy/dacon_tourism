import pandas as pd
import os.path as osp
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

PATH_BASE = './'
PATH_DATA = osp.join(PATH_BASE, 'data')

df = pd.read_csv(osp.join(PATH_DATA, 'train.csv'))

le = preprocessing.LabelEncoder()
le.fit(df['cat3'].values)
df['cat3'] = le.transform(df['cat3'].values)

le = preprocessing.LabelEncoder()
le.fit(df['cat2'].values)
df['cat2'] = le.transform(df['cat2'].values)

le = preprocessing.LabelEncoder()
le.fit(df['cat1'].values)
df['cat1'] = le.transform(df['cat1'].values)

folds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
df['kfold'] = -1
for i in range(5):
    df_idx, valid_idx = list(folds.split(df.values, df['cat3']))[i]
    valid = df.iloc[valid_idx]

    df.loc[df[df.id.isin(valid.id) == True].index.to_list(), 'kfold'] = i

df.to_csv(osp.join(PATH_DATA, 'train_5fold.csv'), index=False)