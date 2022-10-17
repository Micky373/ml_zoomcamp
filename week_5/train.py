
C = 1.0
n_splits = 5
output =f'model_C=({C}).bin'

# Importing libraries

import pandas as pd
import numpy as np
import sys
import warnings
import pickle

warnings.filterwarnings('ignore')
sys.path.append('..')

from scripts import dataframe as dfr
from scripts import matrix as mx
from scripts import regression as rgr
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import auc, roc_auc_score

# Training the model

df = pd.read_csv('../week_3/churn_data.csv')

df = dfr.data_frame_refining(df)

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

df_full_train,df_test = train_test_split(df,test_size=0.2,random_state=1)
df_train,df_val = train_test_split(df_full_train,test_size=0.25,random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

numerical_cols = ['tenure', 'monthlycharges', 'totalcharges']
categorical_cols = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

print(f'Doing the validation with C={C}')

kfold = KFold(n_splits=5,shuffle=True,random_state = 1)
scores = []
fold = 0

for train_idx , val_idx in kfold.split(df_full_train):

    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv , model = dfr.train(df_train,y_train,categorical_cols,numerical_cols,C)
    y_pred = dfr.predict(df_val,categorical_cols,numerical_cols,dv,model)

    auc = roc_auc_score(y_val,y_pred)
    scores.append(auc)
    print(f'auc on fold {fold} is {auc}')
    fold += 1

print('Validation result:')
print(f'C={C} +- mean_score={np.mean(scores)} +- std_score={np.std(scores)}')

# Training the final model

print('Training the final model')
dv,model = dfr.train(df_full_train,df_full_train.churn.values,categorical_cols,numerical_cols,C)
y_pred_final = dfr.predict(df_test,categorical_cols,numerical_cols,dv,model)
auc = roc_auc_score(y_test,y_pred_final)
print(f'auc = {auc}')

# # Saving the model

with open(output,'wb') as f_out:
    pickle.dump((dv,model),f_out)

print(f'The model is saved to {output}')

