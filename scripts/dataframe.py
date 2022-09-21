import numpy as np

def col_name_refining(df,list_):
    for col in list_:
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
