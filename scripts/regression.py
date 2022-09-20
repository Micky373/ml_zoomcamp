import numpy as np

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