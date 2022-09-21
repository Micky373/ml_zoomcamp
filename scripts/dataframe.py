
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