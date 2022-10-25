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

split_idx = list(folds.split(df.values, df['cat3']))

## ver 1
df['kfold'] = -1
for i in range(5):
    df_idx, valid_idx = split_idx[i]
    valid = df.iloc[valid_idx]

    condition = df.id.isin(valid.id) == True
    df.loc[df[condition].index.to_list(), 'kfold'] = i

df.to_csv(osp.join(PATH_DATA, 'train_5fold.csv'), index=False)

## ver 2
min_standard = df['cat3'].value_counts()[df['cat3'].value_counts() >= 5].index.tolist()
df['kfold'] = -1
for i in range(5):
    df_idx, valid_idx = split_idx[i]
    valid = df.iloc[valid_idx]

    condition = ( df.id.isin(valid.id) ) & ( df.cat3.isin(min_standard) )
    df.loc[df[condition].index.to_list(), 'kfold'] = i

df.to_csv(osp.join(PATH_DATA, 'train_5fold_ver2.csv'), index=False)