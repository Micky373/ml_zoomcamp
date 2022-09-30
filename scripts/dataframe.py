import numpy as np
from IPython.display import display
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns 

def data_frame_refining(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    cat_columns = df.dtypes[df.dtypes == 'object'].index
    for col in cat_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    return df

def checking_df(df):
    for col in df.columns:
        print(col)
        print(df[col].unique()[:5])
        print(df[col].nunique())
        print('****')

def train_val_test_split_with_shuffle(df,seed,val_split,test_split,del_col):

    
    n = len(df)
    idx = np.arange(n)

    n_val = int(len(df) * val_split)
    n_test = int(len(df) * test_split)
    n_train = n - (n_val + n_test)

    np.random.seed(seed)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]

    y_train = np.log1p(df_train[del_col].values)
    y_test = np.log1p(df_test[del_col].values)
    y_val = np.log1p(df_val[del_col].values)

    del df_train[del_col]
    del df_test[del_col]
    del df_val[del_col]

    return df_train,df_val,df_test,y_train,y_test,y_val

def display_risk_factor(df,cat_columns,datum_value):
    for col in cat_columns:
        print(col)
        df_group = df.groupby(col).churn.agg(['mean','count'])
        df_group['diff'] = df_group['mean'] - datum_value
        df_group['risk'] = df_group['mean'] / datum_value
        display(df_group)
        print('*******')
        print('*******')

def calculate_mut_score(df,cat_columns,bool_):
    dict_ = {}
    for col in cat_columns:
        dict_[col] = round((mutual_info_score(df[col],df.churn)),5)

    return sorted(dict_.items(), key= lambda x: x[1],reverse=bool_)

def corr_matrix(df,title:str,save_as):
    plt.figure(figsize=(25, 20))
    res=sns.heatmap(df.corr(), annot=True,fmt='.2f');
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 15)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 15)
    plt.title(title,size=18, fontweight='bold')
    plt.savefig(f'../charts/{save_as}')
    plt.show



