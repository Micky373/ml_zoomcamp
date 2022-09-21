import numpy as np
import seaborn as sns

def linear_reg(xi,w0,w):
    pred = w0
    for weight in range(len(w)):
        pred += w[weight]*xi[weight]

    return pred

def linear_reg_vec(xi,w):
    return xi.dot(w)

def train_linear_reg(X,y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones,X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    return XTX_inv.dot(X.T).dot(y)

def train_linear_reg_regularized(X,y,r = 0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones,X])
    XTX = X.T.dot(X)
    XTX = XTX + r*np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    return XTX_inv.dot(X.T).dot(y)

def rmse(y,y_pred):

   error_ = y - y_pred

   se = error_ ** 2
   mse = se.mean()
   return np.sqrt(mse)

def plot_pred(y,y_pred):
    sns.histplot(y_pred,bins=50, alpha = 0.5, color = 'red')
    sns.histplot(y,bins=50, alpha = 0.5, color = 'blue')

def prepare_X(df,base,categories):
    df = df.copy()
    features = base.copy()
    df['age'] = 2017 - df.year
    features.append('age')
    for value in [2,3,4]:
        df['num_of_doors_%s' %value] = (df.number_of_doors == value).astype('int')
        features.append('num_of_doors_%s' %value)
    for category,values in categories.items():

        for value in values:
            df['%s_%s' %(category,value)] = (df[category]== value).astype('int')
            features.append('%s_%s' %(category,value))
    return df[features].fillna(0).values


def show_training_results_with_regularization(X_train,X_val,y_train,y_val,r_list):
    for r in r_list:
        w_all= train_linear_reg_regularized(X_train,y_train,r)
        w0 = w_all[0]
        w = w_all[1:]
        y_val_pred = w0 + X_val.dot(w)
        score = rmse(y_val,y_val_pred)
        print(r, w0, round(score,2))